import streamlit as st
import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from model import CNNGRU




# ==========================
# Matplotlib 中文显示设置
# ==========================
plt.rcParams["font.sans-serif"] = ["SimHei"]      # 黑体
plt.rcParams["axes.unicode_minus"] = False        # 解决负号显示问题

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
    **纯数据驱动 CNN–GRU 温度预测模型**

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
        st.warning("请至少选择一个距入口断面位置。")
        st.stop()

    if len(Axial_Distances) > 4:
        st.error("最多只能选择 4 个断面位置。")
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
        df = pd.DataFrame({"Date": dates, "Temp": y_daily})
        df["Month"] = df["Date"].dt.month
        monthly = df.groupby("Month")["Temp"].mean()

        ax1.plot(
            monthly.index,
            monthly.values,
            marker="o",
            linewidth=2,
            label=f"距入口 {int(dist)} m"
        )

    ax1.set_xlabel("月份")
    ax1.set_ylabel("温度（℃）")
    ax1.set_title("不同隧道断面月平均温度变化")
    ax1.set_xticks(np.arange(1, 13))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(title="断面位置")

    st.pyplot(fig1)

    # ========= 图 2：温度–距离（20 m 间隔，固定月份） =========
    st.markdown("### 沿隧道轴线的温度空间分布")

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
        df = pd.DataFrame({"Date": dates, "Temp": y_daily})
        df["Month"] = df["Date"].dt.month

        month_mean = df[df["Month"] == Selected_Month]["Temp"].mean()
        temps_along_tunnel.append(month_mean)

    temps_along_tunnel = np.array(temps_along_tunnel)

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(
        axial_positions,
        temps_along_tunnel,
        linewidth=2
    )

    # —— 0 ℃ 等温位置标注 ——
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

        ax2.text(
            x0,
            0,
            f"  0 ℃ 对应位置：{x0:.0f} m",
            color="red",
            verticalalignment="bottom"
        )

    ax2.set_xlabel("距入口距离（m）")
    ax2.set_ylabel("温度（℃）")
    ax2.set_title(f"{Selected_Month} 月沿隧道轴线的温度分布")
    ax2.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig2)