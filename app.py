import streamlit as st
import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request

from model import CNNGRU

# ==========================
# Matplotlib 中文字体加载（兼容所有部署环境）
# ==========================

def _get_chinese_font() -> fm.FontProperties | None:
    """
    按优先级尝试获取可用的中文 FontProperties：
    1. 系统已安装的常见中文字体
    2. 项目目录下预置的 .ttf 文件（NotoSansSC）
    3. 自动从 Google Fonts CDN 下载 NotoSansSC-Regular.ttf
    返回 FontProperties 对象；若全部失败则返回 None（回退英文）。
    """
    # ── 1. 扫描系统字体 ──────────────────────────────────────────────
    preferred = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Heiti SC", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
        "Noto Sans SC", "AR PL UMing CN", "KaiTi", "FangSong",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return fm.FontProperties(family=name)

    # ── 2. 项目目录或脚本同级目录下是否有字体文件 ─────────────────────
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
    ]
    font_filenames = [
        "NotoSansSC-Regular.ttf",
        "NotoSansSC.ttf",
        "wqy-microhei.ttc",
        "simhei.ttf",
        "msyh.ttc",
    ]
    for d in search_dirs:
        for fname in font_filenames:
            fpath = os.path.join(d, fname)
            if os.path.isfile(fpath):
                return fm.FontProperties(fname=fpath)

    # ── 3. 自动下载 NotoSansSC-Regular.ttf（约 5 MB，只下载一次） ───
    download_path = os.path.join(os.getcwd(), "NotoSansSC-Regular.ttf")
    cdn_url = (
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/"
        "NotoSansSC-Regular.otf"
    )
    # 备用镜像（jsDelivr）
    cdn_fallback = (
        "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/"
        "Sans/SubsetOTF/SC/NotoSansSC-Regular.otf"
    )
    for url in (cdn_url, cdn_fallback):
        try:
            urllib.request.urlretrieve(url, download_path)
            fm.fontManager.addfont(download_path)
            return fm.FontProperties(fname=download_path)
        except Exception:
            continue

    return None  # 最终失败，回退到默认字体（英文）


@st.cache_resource(show_spinner="正在加载中文字体…")
def get_font() -> fm.FontProperties | None:
    fp = _get_chinese_font()
    if fp is not None:
        # 同步设置全局 rcParams，使 tick labels 等也生效
        plt.rcParams["font.family"] = fp.get_family()
        name = fp.get_name()
        if name:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False
    return fp


CN_FONT = get_font()


def cn(size: int = 11) -> dict:
    """返回可直接传给 matplotlib 文字方法的 fontproperties + size 字典。"""
    if CN_FONT is not None:
        fp = fm.FontProperties(fname=CN_FONT.get_file() or "", family=CN_FONT.get_family())
        fp.set_size(size)
        return {"fontproperties": fp}
    return {"fontsize": size}


def set_cn_ticks(ax, size: int = 10):
    """对坐标轴刻度标签应用中文字体。"""
    if CN_FONT is not None:
        fp = fm.FontProperties(fname=CN_FONT.get_file() or "", family=CN_FONT.get_family())
        fp.set_size(size)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(fp)


# ==========================
# 1. 页面与缓存设置
# ==========================
st.set_page_config(
    page_title="寒区隧道温度场预测",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = CNNGRU(input_dim=7)
    model.load_state_dict(
        torch.load("cnn_gru_temperature_model.pth", map_location="cpu")
    )
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ==========================
# 2. 页面标题
# ==========================
st.title("寒区隧道温度场预测")
st.markdown(
    """
    - 不同隧道断面处的月平均温度演化特征  
    - 沿隧道轴线方向的温度空间分布特征
    """
)

# ==========================
# 3. 侧边栏输入参数（工程合理范围）
# ==========================
st.sidebar.header("输入参数设置")

Tunnel_Length = st.sidebar.number_input(
    "隧道长度（m）",
    min_value=2000.0,
    max_value=10000.0,
    value=2000.0,
    step=100.0
)

distance_candidates = np.arange(
    500,
    int(Tunnel_Length) + 1,
    500
).tolist()

Axial_Distances = st.sidebar.multiselect(
    "距入口断面位置（用于月变化曲线，m）",
    options=distance_candidates,
    default=distance_candidates[:3],
    help="最多选择 4 个断面位置"
)

Selected_Month = st.sidebar.selectbox(
    "选择空间分布分析月份",
    options=list(range(1, 13)),
    format_func=lambda x: f"{x} 月",
    index=0
)

Overburden = st.sidebar.number_input(
    "平均埋置深度（m）",
    min_value=50.0,
    max_value=250.0,
    value=150.0,
    step=10.0
)

Air_Velocity = st.sidebar.number_input(
    "入口风速（m/s）",
    min_value=0.5,
    max_value=2.5,
    value=1.5,
    step=0.1
)

Mean_T = st.sidebar.number_input(
    "年平均气温（℃）",
    min_value=-2.0,
    max_value=6.0,
    value=-2.0,
    step=0.5
)

Amp_T = st.sidebar.number_input(
    "年最大温差（℃）",
    min_value=16.0,
    max_value=24.0,
    value=16.0,
    step=1.0
)

Rock_k = st.sidebar.number_input(
    "围岩导热系数（W/m·K）",
    min_value=1.5,
    max_value=3.5,
    value=2.5,
    step=0.1
)

# ==========================
# 4. 预测与绘图
# ==========================
if st.button("开始预测"):

    # ========= 图 1：月平均温度–月份（多断面） =========
    if len(Axial_Distances) == 0:
        st.warning("请至少选择一个断面位置。")
        st.stop()

    if len(Axial_Distances) > 4:
        st.error("最多选择 4 个断面位置。")
        st.stop()

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    for dist in Axial_Distances:

        features = np.array([
            dist,
            Tunnel_Length,
            Overburden,
            Air_Velocity,
            Mean_T,
            Amp_T,
            Rock_k
        ], dtype=float)

        X = np.tile(features, (365, 1))
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            y_daily = model(X_tensor).numpy().flatten()

        dates = pd.date_range("2024-01-01", periods=365)
        df = pd.DataFrame({"日期": dates, "温度": y_daily})
        df["月份"] = df["日期"].dt.month
        monthly = df.groupby("月份")["温度"].mean()

        ax1.plot(
            monthly.index,
            monthly.values,
            marker="o",
            linewidth=2,
            label=f"距入口 {dist} m"
        )

    ax1.set_xlabel("月份", fontsize=12, **cn(12))
    ax1.set_ylabel("温度（℃）", fontsize=12, **cn(12))
    ax1.set_title("不同断面月平均温度变化曲线", fontsize=14, **cn(14))
    ax1.set_xticks(np.arange(1, 13))
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 图例中文字体
    if CN_FONT is not None:
        legend_fp = fm.FontProperties(
            fname=CN_FONT.get_file() or "",
            family=CN_FONT.get_family(),
            size=10
        )
        title_fp = fm.FontProperties(
            fname=CN_FONT.get_file() or "",
            family=CN_FONT.get_family(),
            size=10
        )
        leg = ax1.legend(
            title="断面位置",
            loc="best",
            prop=legend_fp,
        )
        leg.set_title("断面位置", prop=title_fp)
    else:
        ax1.legend(title="断面位置", loc="best", fontsize=10, title_fontsize=10)

    set_cn_ticks(ax1, size=10)
    fig1.tight_layout()
    st.pyplot(fig1)

    # ========= 图 2：温度–距离（20 m 间隔，固定月份） =========
    st.markdown("### 温度—距离分布")

    axial_positions = np.arange(0, int(Tunnel_Length) + 1, 20)
    temps_along_tunnel = []

    for dist in axial_positions:

        features = np.array([
            dist,
            Tunnel_Length,
            Overburden,
            Air_Velocity,
            Mean_T,
            Amp_T,
            Rock_k
        ], dtype=float)

        X = np.tile(features, (365, 1))
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            y_daily = model(X_tensor).numpy().flatten()

        dates = pd.date_range("2024-01-01", periods=365)
        df = pd.DataFrame({"日期": dates, "温度": y_daily})
        df["月份"] = df["日期"].dt.month

        month_mean = df[df["月份"] == Selected_Month]["温度"].mean()
        temps_along_tunnel.append(month_mean)

    temps_along_tunnel = np.array(temps_along_tunnel)

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(
        axial_positions,
        temps_along_tunnel,
        linewidth=2
    )

    # —— 0 ℃ 位置标注 ——
    sign_change_idx = np.where(np.diff(np.sign(temps_along_tunnel)))[0]
    if len(sign_change_idx) > 0:
        idx = sign_change_idx[0]
        x0 = np.interp(
            0,
            [temps_along_tunnel[idx], temps_along_tunnel[idx + 1]],
            [axial_positions[idx], axial_positions[idx + 1]]
        )

        ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax2.axvline(x0, color="red", linestyle="--", linewidth=1)
        ax2.scatter(x0, 0, color="red", zorder=5)

        anno_kwargs = {"fontsize": 9, "color": "red"}
        if CN_FONT is not None:
            anno_fp = fm.FontProperties(
                fname=CN_FONT.get_file() or "",
                family=CN_FONT.get_family(),
                size=9
            )
            anno_kwargs = {"fontproperties": anno_fp, "color": "red"}

        ax2.annotate(
            f"0℃ 位置：{x0:.0f} m",
            xy=(x0, 0),
            xytext=(x0 + max(axial_positions) * 0.03, 0.5),
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
            **anno_kwargs
        )

    ax2.set_xlabel("距入口轴向距离（m）", **cn(12))
    ax2.set_ylabel("温度（℃）", **cn(12))
    ax2.set_title(f"隧道轴线温度分布（{Selected_Month} 月）", **cn(14))
    ax2.grid(True, linestyle="--", alpha=0.6)

    set_cn_ticks(ax2, size=10)
    fig2.tight_layout()
    st.pyplot(fig2)
