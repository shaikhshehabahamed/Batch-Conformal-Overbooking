"""Generates supplementary figures from saved analysis outputs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from batch_risk_control.plotting import (
    setup_style, clean_spines, save_fig, METHOD_COLORS, METHOD_LABELS,
    METHOD_MARKERS, INK_MUTED, INK_SECONDARY, TARGET_LINE,
)

DATA = ROOT / "data" / "processed"
OUT = ROOT / "paper_assets" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
setup_style()
SEQ_BLUE = "#2a78d6"


def fig_s1_eda_neighbourhood():
    df = pd.read_csv(DATA / "noshow_eda_neighbourhood.csv").sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(df["Neighbourhood"], df["mean"], color=SEQ_BLUE, height=0.6)
    ax.set_xlabel("Attendance rate")
    ax.set_xlim(0, 1)
    ax.axvline(df["mean"].mean(), color=TARGET_LINE, ls="--", lw=1.0)
    clean_spines(ax)
    ax.set_title("Attendance rate by clinic (top-10 by volume)", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, str(OUT / "figS1_eda_neighbourhood"))
    plt.close(fig)
    print("Saved Figure S1")


def fig_s2_eda_hotel_seasonality():
    df = pd.read_csv(DATA / "hotel_eda_monthly.csv")
    df["arrival_date"] = pd.PeriodIndex(df["arrival_date"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(df["arrival_date"], df["mean"], color=SEQ_BLUE, lw=1.8, marker="o", ms=3.5)
    ax.set_ylabel("Fulfillment rate")
    ax.set_xlabel("Arrival month")
    clean_spines(ax)
    ax.set_title("Hotel fulfillment rate by month (2015-07 to 2017-08): non-monotonic seasonality",
                 loc="left", fontweight="bold", fontsize=10)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    save_fig(fig, str(OUT / "figS2_eda_hotel_seasonality"))
    plt.close(fig)
    print("Saved Figure S2")


def fig_s3_alpha_sensitivity():
    df = pd.read_csv(DATA / "simulation_study_raw.csv")
    df = df[df["config"].str.startswith("alpha_")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    alphas = sorted(df["alpha"].unique())

    for method in ["ours", "bonferroni", "dro_plugin", "dro_calib", "naive"]:
        sub = df[df["method"] == method]
        agg = sub.groupby("alpha")["coverage"].mean()
        target = 1 - agg.index
        axes[0].plot(agg.index, agg.values, marker=METHOD_MARKERS[method], ms=5, lw=1.6,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        agg_w = sub.groupby("alpha")["width"].mean()
        axes[1].plot(agg_w.index, agg_w.values, marker=METHOD_MARKERS[method], ms=5, lw=1.6,
                     color=METHOD_COLORS[method])

    axes[0].plot(alphas, [1 - a for a in alphas], color=TARGET_LINE, ls="--", lw=1.0, zorder=0, label="Nominal target")
    axes[0].set_xlabel(r"Miscoverage level ($\alpha$)")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("(a) Coverage", loc="left", fontweight="bold", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8)
    clean_spines(axes[0])

    axes[1].set_xlabel(r"Miscoverage level ($\alpha$)")
    axes[1].set_ylabel("Mean interval width")
    axes[1].set_title("(b) Width", loc="left", fontweight="bold", fontsize=10)
    clean_spines(axes[1])

    fig.suptitle(r"Sensitivity to $\alpha$ at $\tau=1.2$ (50 replications per point)",
                 fontsize=10, y=1.03, color=INK_MUTED)
    fig.tight_layout()
    save_fig(fig, str(OUT / "figS3_alpha_sensitivity"))
    plt.close(fig)
    print("Saved Figure S3")


def fig_s4_ablation():
    df = pd.read_csv(DATA / "calibration_ablation.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = [INK_SECONDARY, "#9ec5f4", "#5598e7", METHOD_COLORS["ours"]]
    bars = ax.bar(df["method"], df["coverage"], color=colors, width=0.6, edgecolor="white", linewidth=1.5)
    ax.axhline(0.90, color=TARGET_LINE, ls="--", lw=1.0, zorder=0)
    ax.text(3.5, 0.915, "90% target", ha="right", fontsize=8, color=INK_MUTED)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.05)
    clean_spines(ax)
    ax.tick_params(axis="x", rotation=18)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_title("Calibration ablation: incremental fixes vs. the adopted ACI fix",
                 loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, str(OUT / "figS4_calibration_ablation"))
    plt.close(fig)
    print("Saved Figure S4")


def fig_s5_classifier_robustness():
    df = pd.read_csv(DATA / "classifier_robustness.csv")
    methods = ["naive", "dro_plugin", "dro_calib", "ours_static", "ours_ACI", "bonferroni"]
    classifiers = df["classifier"].unique()
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, clf in enumerate(classifiers):
        sub = df[df["classifier"] == clf].set_index("method").reindex(methods)
        offset = (i - 0.5) * width
        hatch = None if i == 0 else "///"
        bars = ax.bar(x + offset, sub["coverage"], width=width,
                      color=[METHOD_COLORS.get(m, "#888") for m in methods],
                      edgecolor="white", linewidth=1.2, hatch=hatch,
                      label=clf.replace("_", " ").title())
    ax.axhline(0.90, color=TARGET_LINE, ls="--", lw=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=25, ha="right")
    ax.set_ylabel("Empirical coverage")
    clean_spines(ax)

    # legend distinguishing classifier by fill pattern (color already means method)
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="white", edgecolor=INK_SECONDARY, label="Logistic regression"),
                    Patch(facecolor="white", edgecolor=INK_SECONDARY, hatch="///", label="Gradient boosting")]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=8.5)
    ax.set_title("Classifier robustness: logistic regression vs. gradient boosting (temporal split)",
                 loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, str(OUT / "figS5_classifier_robustness"))
    plt.close(fig)
    print("Saved Figure S5")


def fig_s6_aci_gamma_sweep():
    df = pd.read_csv(DATA / "noshow_aci_gamma_sweep.csv")
    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax1.plot(df["gamma"], df["coverage"], marker="o", ms=6, lw=1.8, color=METHOD_COLORS["ours"])
    ax1.axhline(0.90, color=TARGET_LINE, ls="--", lw=1.0, zorder=0)
    ax1.set_xlabel(r"ACI step size ($\gamma$)")
    ax1.set_ylabel("Empirical coverage", color=METHOD_COLORS["ours"])
    ax1.tick_params(axis="y", labelcolor=METHOD_COLORS["ours"])
    clean_spines(ax1)
    for _, row in df.iterrows():
        ax1.annotate(f"{row['width']:.1f}", xy=(row["gamma"], row["coverage"]),
                     xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK_MUTED)
    ax1.set_title(r"ACI: coverage vs. $\gamma$ (label = mean width; $\gamma$=0.02 chosen)",
                  loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, str(OUT / "figS6_aci_gamma_sweep"))
    plt.close(fig)
    print("Saved Figure S6")


def fig_s7_decision_layer_capacity():
    df = pd.read_csv(DATA / "decision_layer_real_data.csv")
    methods = ["naive", "dro_plugin", "dro_calib", "bonferroni", "ours_ACI"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for method in methods:
        sub = df[df["method"] == method].sort_values("capacity_fraction")
        marker = METHOD_MARKERS.get(method, "o")
        color = METHOD_COLORS.get(method, "#888")
        axes[0].plot(sub["capacity_fraction"], sub["overflow_rate"], marker=marker, ms=6, lw=1.8,
                     color=color, label=METHOD_LABELS.get(method, method))
        axes[1].plot(sub["capacity_fraction"], sub["cost"], marker=marker, ms=6, lw=1.8, color=color)
    axes[0].axhline(0.10, color=TARGET_LINE, ls="--", lw=1.0, zorder=0)
    axes[0].set_xlabel("Capacity (fraction of historical demand)")
    axes[0].set_ylabel("Overflow rate")
    axes[0].set_title("(a) Overflow rate", loc="left", fontweight="bold", fontsize=10)
    clean_spines(axes[0])
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Capacity (fraction of historical demand)")
    axes[1].set_ylabel("Mean realized cost")
    axes[1].set_title("(b) Cost", loc="left", fontweight="bold", fontsize=10)
    clean_spines(axes[1])

    fig.suptitle("Real-data decision layer across capacity levels (medical no-show dataset)",
                 fontsize=10, y=1.03, color=INK_MUTED)
    fig.tight_layout()
    save_fig(fig, str(OUT / "figS7_decision_layer_capacity"))
    plt.close(fig)
    print("Saved Figure S7")


if __name__ == "__main__":
    fig_s1_eda_neighbourhood()
    fig_s2_eda_hotel_seasonality()
    fig_s3_alpha_sensitivity()
    fig_s4_ablation()
    fig_s5_classifier_robustness()
    fig_s6_aci_gamma_sweep()
    fig_s7_decision_layer_capacity()
    print(f"\nAll supplementary figures saved to {OUT}")
