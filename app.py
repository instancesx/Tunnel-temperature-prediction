import streamlit as st
import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from model import CNNGRU

# ==========================
# Matplotlib中文显示设置
# ==========================
matplotlib.rcParams["font.family"] = "sans-serif"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS"
]

matplotlib.rcParams["axes.unicode_minus"] = False

# ==========================
# 页面设置
# ==========================
st.set_page_config(
    page_title="寒区隧道温度场预测平台",
    layout="centered"
)

# ==========================
# 模型加载
# ==========================
@st.cache_resource
def load_model():

    model = CNNGRU(input_dim=7)

    model.load_state_dict(
        torch.load(
            "cnn_gru_temperature_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")


model = load_model()
scaler = load_scaler()

# ==========================
# 页面标题
# ==========================
st.title("寒区隧道温度场预测平台")

st.markdown(
    """
### CNN-GRU温度场预测模型

本平台基于深度学习模型实现寒区隧道温度场预测，可分析：

- 不同断面月平均温度变化规律
- 沿隧道轴线方向温度空间分布特征
- 冻结界面（0℃等温线）位置变化规律
"""
)

# ==========================
# 侧边栏参数输入
# ==========================
st.sidebar.header("参数设置")

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
    "选择断面位置（m）",
    options=distance_candidates,
    default=distance_candidates[:3],
    help="最多选择4个断面"
)

Selected_Month = st.sidebar.selectbox(
    "空间分布分析月份",
    options=list(range(1, 13)),
    format_func=lambda x: f"{x} 月",
    index=0
)

Overburden = st.sidebar.number_input(
    "平均埋深（m）",
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
    "年温度振幅（℃）",
    min_value=16.0,
    max_value=24.0,
    value=16.0,
    step=1.0
)

Rock_k = st.sidebar.number_input(
    "围岩导热系数（W/(m·K)）",
    min_value=1.5,
    max_value=3.5,
    value=2.5,
    step=0.1
)

# ==========================
# 开始预测
# ==========================
if st.button("开始预测"):

    if len(Axial_Distances) == 0:
        st.warning("请至少选择一个断面位置")
        st.stop()

    if len(Axial_Distances) > 4:
        st.error("最多选择4个断面位置")
        st.stop()

    # ==================================================
    # 图1 月平均温度变化曲线
    # ==================================================
    fig1, ax1 = plt.subplots(figsize=(9, 5))

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

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            y_daily = model(
                X_tensor
            ).numpy().flatten()

        dates = pd.date_range(
            "2024-01-01",
            periods=365
        )

        df = pd.DataFrame({
            "日期": dates,
            "温度": y_daily
        })

        df["月份"] = df["日期"].dt.month

        monthly = df.groupby(
            "月份"
        )["温度"].mean()

        ax1.plot(
            monthly.index,
            monthly.values,
            marker="o",
            linewidth=2.5,
            label=f"距洞口 {dist} m"
        )

    ax1.set_xlabel(
        "月份",
        fontsize=12
    )

    ax1.set_ylabel(
        "温度（℃）",
        fontsize=12
    )

    ax1.set_title(
        "不同断面月平均温度变化曲线",
        fontsize=14,
        fontweight="bold"
    )

    ax1.set_xticks(
        np.arange(1, 13)
    )

    ax1.legend(
        title="断面位置"
    )

    ax1.grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    st.pyplot(fig1)

    # ==================================================
    # 图2 温度-距离分布
    # ==================================================
    st.markdown("## 隧道轴向温度分布")

    axial_positions = np.arange(
        0,
        int(Tunnel_Length) + 1,
        20
    )

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

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            y_daily = model(
                X_tensor
            ).numpy().flatten()

        dates = pd.date_range(
            "2024-01-01",
            periods=365
        )

        df = pd.DataFrame({
            "日期": dates,
            "温度": y_daily
        })

        df["月份"] = df["日期"].dt.month

        month_mean = df[
            df["月份"] == Selected_Month
        ]["温度"].mean()

        temps_along_tunnel.append(
            month_mean
        )

    temps_along_tunnel = np.array(
        temps_along_tunnel
    )

    fig2, ax2 = plt.subplots(
        figsize=(9, 5)
    )

    ax2.plot(
        axial_positions,
        temps_along_tunnel,
        linewidth=2.5
    )

    # ==========================
    # 0℃冻结界面识别
    # ==========================
    sign_change_idx = np.where(
        np.diff(
            np.sign(
                temps_along_tunnel
            )
        )
    )[0]

    if len(sign_change_idx) > 0:

        idx = sign_change_idx[0]

        x0 = np.interp(
            0,
            [
                temps_along_tunnel[idx],
                temps_along_tunnel[idx + 1]
            ],
            [
                axial_positions[idx],
                axial_positions[idx + 1]
            ]
        )

        ax2.axhline(
            0,
            color="gray",
            linestyle="--"
        )

        ax2.axvline(
            x0,
            color="red",
            linestyle="--"
        )

        ax2.scatter(
            x0,
            0,
            color="red",
            s=60,
            label=f"0℃位置：{x0:.0f} m"
        )

        ax2.legend()

    ax2.set_xlabel(
        "距洞口轴向距离（m）",
        fontsize=12
    )

    ax2.set_ylabel(
        "温度（℃）",
        fontsize=12
    )

    ax2.set_title(
        f"{Selected_Month}月隧道轴向温度分布",
        fontsize=14,
        fontweight="bold"
    )

    ax2.grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    st.pyplot(fig2)