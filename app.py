import os
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from model import CNNGRU


# ==========================
# 0. Matplotlib 中文显示设置
# ==========================
def setup_chinese_font():
    """
    自动设置 Matplotlib 中文字体，避免坐标轴、标题和图例出现方框。

    使用逻辑：
    1. 优先读取项目目录下的中文字体文件，例如 ./fonts/SimHei.ttf、./assets/SimHei.ttf；
    2. 若未提供字体文件，则自动查找系统已安装中文字体；
    3. 若仍未找到，则继续运行，但给出提示，此时图中文字可能显示为方框。
    """
    candidate_font_files = [
        "./SimHei.ttf",
        "./Microsoft YaHei.ttf",
        "./msyh.ttc",
        "./fonts/SimHei.ttf",
        "./fonts/Microsoft YaHei.ttf",
        "./fonts/msyh.ttc",
        "./assets/SimHei.ttf",
        "./assets/Microsoft YaHei.ttf",
        "./assets/msyh.ttc",
    ]

    for font_file in candidate_font_files:
        font_path = Path(font_file)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    candidate_font_names = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "KaiTi",
        "FangSong",
    ]

    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_font_names:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    plt.rcParams["axes.unicode_minus"] = False
    return None


CHINESE_FONT = setup_chinese_font()


# ==========================
# 1. 页面与缓存设置
# ==========================
st.set_page_config(
    page_title="寒区隧道温度场预测平台",
    layout="centered"
)

# Streamlit 页面本身的中文字体设置。该部分只影响网页文字，不影响 Matplotlib 图中文字。
st.markdown(
    """
    <style>
    html, body, [class*="css"], [class*="st-"] {
        font-family: "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
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
st.title("寒区隧道温度场预测平台")
st.markdown(
    """
    **纯数据驱动 CNN–GRU 温度预测模型**

    - 不同隧道断面处的月平均温度演化特征  
    - 沿隧道轴线方向的温度空间分布特征
    """
)

if CHINESE_FONT is None:
    st.warning(
        "当前运行环境未检测到可用中文字体，图中的中文可能显示为方框。"
        "建议在项目目录下新建 fonts 文件夹，并放入 SimHei.ttf 或 Microsoft YaHei.ttf。"
    )


# ==========================
# 3. 侧边栏输入参数
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
# 4. 通用预测函数
# ==========================
def predict_daily_temperature(dist):
    """根据断面距入口距离及输入参数，预测 365 天日平均温度。"""
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

    return y_daily


# ==========================
# 5. 预测与绘图
# ==========================
if st.button("开始预测"):

    # ========= 图 1：月平均温度—月份（多断面） =========
    if len(Axial_Distances) == 0:
        st.warning("请至少选择一个断面位置。")
        st.stop()

    if len(Axial_Distances) > 4:
        st.error("最多选择 4 个断面位置。")
        st.stop()

    st.markdown("### 不同断面月平均温度变化")

    fig1, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=150)

    for dist in Axial_Distances:
        y_daily = predict_daily_temperature(dist)

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

    ax1.set_xlabel("月份", fontsize=12)
    ax1.set_ylabel("温度（℃）", fontsize=12)
    ax1.set_title("不同断面月平均温度变化曲线", fontsize=14, pad=12)
    ax1.set_xticks(np.arange(1, 13))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(title="断面位置", loc="best", fontsize=10, title_fontsize=10)
    fig1.tight_layout()

    st.pyplot(fig1, clear_figure=True)

    # ========= 图 2：温度—距离（20 m 间隔，固定月份） =========
    st.markdown("### 隧道轴线温度分布")

    axial_positions = np.arange(0, int(Tunnel_Length) + 1, 20)
    temps_along_tunnel = []

    for dist in axial_positions:
        y_daily = predict_daily_temperature(dist)

        dates = pd.date_range("2024-01-01", periods=365)
        df = pd.DataFrame({"日期": dates, "温度": y_daily})
        df["月份"] = df["日期"].dt.month

        month_mean = df.loc[df["月份"] == Selected_Month, "温度"].mean()
        temps_along_tunnel.append(month_mean)

    temps_along_tunnel = np.array(temps_along_tunnel)

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.6), dpi=150)

    ax2.plot(
        axial_positions,
        temps_along_tunnel,
        linewidth=2,
        label=f"{Selected_Month} 月平均温度"
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
        ax2.annotate(
            f"0 ℃ 位置：{x0:.0f} m",
            xy=(x0, 0),
            xytext=(x0 + max(axial_positions) * 0.03, 0.5),
            fontsize=9,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1)
        )

    ax2.set_xlabel("距入口轴向距离（m）", fontsize=12)
    ax2.set_ylabel("温度（℃）", fontsize=12)
    ax2.set_title(f"隧道轴线温度分布（{Selected_Month} 月）", fontsize=14, pad=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best", fontsize=10)
    fig2.tight_layout()

    st.pyplot(fig2, clear_figure=True)
