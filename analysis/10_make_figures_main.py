"""Generates the four main paper figures from the saved analysis outputs.
Reads only data/processed/*.csv -- never re-runs an experiment, matching
the "figures are a pure function of saved results" separation of concerns.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from batch_risk_control.plotting import (
    setup_style, clean_spines, save_fig, METHOD_COLORS, METHOD_LABELS,
    METHOD_ORDER, METHOD_MARKERS, PERIOD_COLORS, INK_MUTED, TARGET_LINE,
)

DATA = ROOT / "data" / "processed"
OUT_MAIN = ROOT / "paper_assets" / "figures" / "main"
OUT_MAIN.mkdir(parents=True, exist_ok=True)
setup_style()


# ---------------------------------------------------------------------
# Figure 1: Simulation validation -- coverage & width vs. tau
# ---------------------------------------------------------------------
def figure1():
    df = pd.read_csv(DATA / "simulation_study_raw.csv")
    df = df[df["config"].str.startswith("tau_")]
    taus = sorted(df["tau"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        agg = sub.groupby("tau")["coverage"].agg(["mean", "std", "count"])
        se = agg["std"] / np.sqrt(agg["count"])
        axes[0].plot(agg.index, agg["mean"], marker=METHOD_MARKERS[method], ms=5, lw=1.6,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[0].fill_between(agg.index, agg["mean"] - 1.96 * se, agg["mean"] + 1.96 * se,
                              color=METHOD_COLORS[method], alpha=0.15, linewidth=0)

        agg_w = sub.groupby("tau")["width"].agg(["mean", "std", "count"])
        se_w = agg_w["std"] / np.sqrt(agg_w["count"])
        axes[1].plot(agg_w.index, agg_w["mean"], marker=METHOD_MARKERS[method], ms=5, lw=1.6,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].fill_between(agg_w.index, agg_w["mean"] - 1.96 * se_w, agg_w["mean"] + 1.96 * se_w,
                              color=METHOD_COLORS[method], alpha=0.15, linewidth=0)

    axes[0].axhline(0.90, color=TARGET_LINE, ls="--", lw=1.0, zorder=0)
    axes[0].text(taus[-1], 0.905, "90% target", ha="right", va="bottom", fontsize=8, color=INK_MUTED)
    axes[0].set_xlabel(r"Within-batch correlation ($\tau$)")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0.55, 1.03)
    axes[0].set_title("(a) Coverage", loc="left", fontweight="bold", fontsize=10)
    clean_spines(axes[0])

    axes[1].set_xlabel(r"Within-batch correlation ($\tau$)")
    axes[1].set_ylabel("Mean interval width")
    axes[1].set_title("(b) Width", loc="left", fontweight="bold", fontsize=10)
    clean_spines(axes[1])
    axes[1].legend(loc="upper left", ncol=1, bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    fig.suptitle("Coverage and width across the correlation sweep (50 replications per point, 95% CI shaded)",
                 fontsize=10, y=1.04, color=INK_MUTED)
    fig.tight_layout()
    save_fig(fig, str(OUT_MAIN / "fig1_simulation_coverage_width"))
    plt.close(fig)
    print("Saved Figure 1")


# ---------------------------------------------------------------------
# Figure 2: Real-data validation -- the failure-and-fix story
# ---------------------------------------------------------------------
def _asymmetric_yerr(sub):
    """[lower, upper] error-bar lengths from stored bootstrap CI bounds --
    NaN-safe so a row missing CI columns just gets a zero-length bar rather
    than crashing plotting."""
    lo = (sub["coverage"] - sub["coverage_ci_lo"]).clip(lower=0).fillna(0)
    hi = (sub["coverage_ci_hi"] - sub["coverage"]).clip(lower=0).fillna(0)
    return np.vstack([lo.to_numpy(), hi.to_numpy()])


def figure2():
    df = pd.read_csv(DATA / "noshow_final_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)

    order_a = ["naive", "dro_plugin", "dro_calib", "ours_static", "bonferroni"]
    order_b = ["naive", "dro_plugin", "dro_calib", "ours_static", "ours_ACI", "bonferroni"]
    errbar_kw = dict(capsize=3, ecolor=INK_MUTED, elinewidth=1.0)

    # panel (a): random split, 5 methods
    sub = df[df["split"] == "random"].set_index("method").reindex(order_a)
    labels_a = [METHOD_LABELS.get(m, m) for m in order_a]
    colors_a = [METHOD_COLORS.get(m, "#888888") for m in order_a]
    axes[0].bar(labels_a, sub["coverage"], color=colors_a, width=0.6, edgecolor="white", linewidth=1.5,
                yerr=_asymmetric_yerr(sub), error_kw=errbar_kw)

    # panel (b): temporal split, static baselines + BOTH ours variants as their
    # own clean category (no manual offsets -> no collision risk)
    static_sub = df[df["split"] == "temporal_static"].set_index("method")
    aci_sub = df[(df["split"] == "temporal_ACI")].set_index("method")
    combined = pd.concat([static_sub, aci_sub])
    sub_b = combined.reindex(order_b)
    labels_b = [METHOD_LABELS.get(m, m) for m in order_b]
    colors_b = [METHOD_COLORS.get(m, "#888888") for m in order_b]
    hatches_b = ["///" if m == "ours_ACI" else None for m in order_b]
    bars_b = axes[1].bar(labels_b, sub_b["coverage"], color=colors_b, width=0.6,
                          edgecolor="white", linewidth=1.5,
                          yerr=_asymmetric_yerr(sub_b), error_kw=errbar_kw)
    for bar, h in zip(bars_b, hatches_b):
        if h:
            bar.set_hatch(h)

    for ax, title in [(axes[0], "(a) Random split (favorable)"),
                       (axes[1], "(b) Temporal split (realistic)")]:
        ax.axhline(0.90, color=TARGET_LINE, ls="--", lw=1.0, zorder=0)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.15)
        clean_spines(ax)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    axes[0].set_ylabel("Empirical coverage")
    fig.suptitle("Medical no-show dataset: static calibration fails under real temporal drift, ACI restores it",
                 fontsize=10, y=1.04, color=INK_MUTED)
    fig.tight_layout()
    save_fig(fig, str(OUT_MAIN / "fig2_real_data_failure_and_fix"))
    plt.close(fig)
    print("Saved Figure 2")


# ---------------------------------------------------------------------
# Figure 3: Decision-layer risk-adjusted value (simulation)
# ---------------------------------------------------------------------
def figure3():
    df = pd.read_csv(DATA / "decision_layer_simulation_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        agg_o = sub.groupby("tau")["overflow_rate"].mean()
        axes[0].plot(agg_o.index, agg_o.values, marker=METHOD_MARKERS[method], ms=5, lw=1.6,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        agg_c = sub.groupby("tau")["cost"].mean()
        axes[1].plot(agg_c.index, agg_c.values, marker=METHOD_MARKERS[method], ms=5, lw=1.6,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    axes[0].axhline(0.10, color=TARGET_LINE, ls="--", lw=1.0, zorder=0)
    axes[0].text(2.0, 0.115, "10% target", ha="right", va="bottom", fontsize=8, color=INK_MUTED)
    axes[0].set_xlabel(r"Within-batch correlation ($\tau$)")
    axes[0].set_ylabel("Overflow rate")
    axes[0].set_title("(a) Overflow rate", loc="left", fontweight="bold", fontsize=10)
    clean_spines(axes[0])

    axes[1].set_xlabel(r"Within-batch correlation ($\tau$)")
    axes[1].set_ylabel("Mean realized cost")
    axes[1].set_title("(b) Cost", loc="left", fontweight="bold", fontsize=10)
    clean_spines(axes[1])
    axes[1].legend(loc="upper left", ncol=1, bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    fig.suptitle("Downstream overbooking decision: naive and plug-in DRO cross into invalid territory as "
                 r"$\tau$ grows; ours does not", fontsize=10, y=1.04, color=INK_MUTED)
    fig.tight_layout()
    save_fig(fig, str(OUT_MAIN / "fig3_decision_layer_simulation"))
    plt.close(fig)
    print("Saved Figure 3")


# ---------------------------------------------------------------------
# Figure 4: Theory validation -- the drift mechanism, visualized
# ---------------------------------------------------------------------
def figure4():
    sys.path.insert(0, str(ROOT / "src"))
    from batch_risk_control.clean_noshow import load_and_clean, build_batches, FEATURE_COLS
    from batch_risk_control.methods import _mu_sigma
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import gaussian_kde

    df_raw = load_and_clean()
    all_batches = build_batches(df_raw)
    dates = sorted(df_raw["AppointmentDay"].dt.date.unique())
    n_d = len(dates)
    train_dates = dates[: int(0.4 * n_d)]
    cal_dates = dates[int(0.4 * n_d): int(0.7 * n_d)]
    test_dates = dates[int(0.7 * n_d):]
    train_df = df_raw[df_raw["AppointmentDay"].dt.date.isin(train_dates)]
    clf = LogisticRegression(max_iter=500)
    clf.fit(train_df[FEATURE_COLS].to_numpy(dtype=float), train_df["Attended"].to_numpy(dtype=int))
    by_date = {}
    for b in all_batches:
        by_date.setdefault(b["appointment_day"].date(), []).append(b)
    cal_batches = [b for d in cal_dates for b in by_date.get(d, [])]
    test_batches = [b for d in test_dates for b in by_date.get(d, [])]

    def compute_R(batches):
        R = []
        for b in batches:
            p_hat = clf.predict_proba(b["X"])[:, 1]
            mu, sigma = _mu_sigma(p_hat)
            R.append((np.sum(b["Y"]) - mu) / sigma)
        return np.array(R)

    R_cal, R_test = compute_R(cal_batches), compute_R(test_batches)
    theory_df = pd.read_csv(DATA / "theory_drift_validation.csv")
    q_hat = theory_df["q_hat"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    xs = np.linspace(min(R_cal.min(), R_test.min()) - 0.5, max(R_cal.max(), R_test.max()) + 0.5, 400)
    kde_cal, kde_test = gaussian_kde(R_cal), gaussian_kde(R_test)
    # Deliberately NOT method colors: this is a time-period contrast, not a
    # method comparison, and reusing e.g. "Bonferroni green" here would
    # silently mean something different than in Figures 1-3.
    ax.fill_between(xs, kde_cal(xs), color=PERIOD_COLORS["before"], alpha=0.20, linewidth=0)
    ax.plot(xs, kde_cal(xs), color=PERIOD_COLORS["before"], lw=1.6, label="Calibration period")
    ax.fill_between(xs, kde_test(xs), color=PERIOD_COLORS["after"], alpha=0.20, linewidth=0)
    ax.plot(xs, kde_test(xs), color=PERIOD_COLORS["after"], lw=1.6, label="Test period")
    ax.axvline(q_hat, color="#0b0b0b", lw=1.2, ls="--")
    ax.annotate(r"calibrated $\hat{q}$", xy=(q_hat, ax.get_ylim()[1] * 0.92),
                xytext=(q_hat + 0.35, ax.get_ylim()[1] * 0.92), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    ax.set_xlabel(r"Nonconformity score $R = (S_b - \hat{\mu}_b)/\hat{\sigma}_b$")
    ax.set_ylabel("Density")
    clean_spines(ax)
    ax.legend(loc="upper left")
    pred, obs = theory_df["predicted_coverage"].iloc[0], theory_df["observed_coverage"].iloc[0]
    ax.set_title(f"Predicted coverage {pred:.1%} vs. observed {obs:.1%} "
                 f"(from calibration-period statistics alone)", fontsize=9.5, loc="left", color=INK_MUTED)
    fig.tight_layout()
    save_fig(fig, str(OUT_MAIN / "fig4_theory_drift_mechanism"))
    plt.close(fig)
    print("Saved Figure 4")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print(f"\nAll main figures saved to {OUT_MAIN}")
