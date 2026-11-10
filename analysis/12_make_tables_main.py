"""Generates the four main paper tables from saved analysis outputs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from batch_risk_control.tables_utils import save_table
from batch_risk_control.plotting import METHOD_LABELS

DATA = ROOT / "data" / "processed"
OUT = ROOT / "paper_assets" / "tables" / "main"

METHOD_DISPLAY_ORDER = ["ours", "bonferroni", "dro_plugin", "dro_calib", "naive"]


def table1_simulation_sweep():
    df = pd.read_csv(DATA / "simulation_study_raw.csv")
    df = df[df["config"].str.startswith("tau_")]
    rows = []
    for tau in sorted(df["tau"].unique()):
        for method in METHOD_DISPLAY_ORDER:
            sub = df[(df["tau"] == tau) & (df["method"] == method)]
            cov_mean, cov_se = sub["coverage"].mean(), sub["coverage"].std(ddof=1) / np.sqrt(len(sub))
            w_mean, w_se = sub["width"].mean(), sub["width"].std(ddof=1) / np.sqrt(len(sub))
            rows.append({
                r"$\tau$": tau, "Method": METHOD_LABELS[method],
                "Coverage": cov_mean, "Coverage 95\\% CI": f"[{cov_mean-1.96*cov_se:.3f}, {cov_mean+1.96*cov_se:.3f}]",
                "Width": w_mean, "Width 95\\% CI": f"[{w_mean-1.96*w_se:.1f}, {w_mean+1.96*w_se:.1f}]",
            })
    out = pd.DataFrame(rows)
    save_table(out, OUT / "table1_simulation_sweep", index=False,
               caption="Coverage and width across the within-batch correlation sweep "
                       "(50 replications per point; nominal target $1-\\alpha=0.90$).",
               label="tab:simulation-sweep",
               column_format="llrlrl",
               notes="95\\% CIs computed via the normal approximation across replications.")
    print("Saved Table 1")


def table2_real_data_comparison():
    rows = []
    for dataset, fname in [("Medical no-show", "noshow_final_comparison.csv"),
                            ("Hotel booking", "hotel_final_comparison.csv")]:
        df = pd.read_csv(DATA / fname)
        for split, split_label in [("random", "Random"), ("temporal_static", "Temporal (static)"),
                                    ("temporal_ACI", "Temporal (ACI)")]:
            sub = df[df["split"] == split]
            for _, r in sub.iterrows():
                rows.append({"Dataset": dataset, "Split": split_label,
                             "Method": METHOD_LABELS.get(r["method"], r["method"]),
                             "Coverage": r["coverage"],
                             "Coverage 95\\% CI": f"[{r['coverage_ci_lo']:.3f}, {r['coverage_ci_hi']:.3f}]",
                             "Width": r["width"],
                             "Width 95\\% CI": f"[{r['width_ci_lo']:.1f}, {r['width_ci_hi']:.1f}]"})
    out = pd.DataFrame(rows)
    save_table(out, OUT / "table2_real_data_comparison", index=False,
               caption="Coverage and width on both real datasets: random split (best case), "
                       "temporal split with static calibration (realistic deployment, exchangeability "
                       "stress test), and temporal split with ACI (the fix).",
               label="tab:real-data-comparison",
               column_format="lllrlrl",
               notes="Nominal target $1-\\alpha=0.90$ throughout. 95\\% CIs are a two-stage "
                     "batch-level bootstrap (1{,}000 replicates) for the static rows and a "
                     "moving-block bootstrap over days (500 replicates) for the ACI rows, "
                     "capturing calibration- and test-sampling variability around each point "
                     "estimate from its single train/cal/test split.")
    print("Saved Table 2")


def table3_decision_layer_real_data():
    df = pd.read_csv(DATA / "decision_layer_real_data.csv")
    df["Method"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    out = df[["capacity_fraction", "Method", "overflow_rate", "mean_utilization", "mean_n_booked", "cost"]].copy()
    out.columns = ["Capacity", "Method", "Overflow rate", "Utilization", "Mean N booked", "Cost"]
    out = out.sort_values(["Capacity", "Method"])
    save_table(out, OUT / "table3_decision_layer_real_data", index=False,
               caption="Downstream overbooking decision on the medical no-show dataset: "
                       "realized outcomes swept across three capacity levels "
                       "(fraction of each batch's historical demand).",
               label="tab:decision-layer-real-data",
               column_format="llrrrr",
               notes="Cost $= 3\\times$ mean overflow amount $+\\ 1\\times$ mean idle slots (arbitrary relative units).")
    print("Saved Table 3")


def table4_theory_validation():
    df = pd.read_csv(DATA / "theory_drift_validation.csv")
    out = pd.DataFrame({
        "Quantity": ["$n$ (calibration)", "$n$ (test)", "Mean $R$ (calibration)", "Mean $R$ (test)",
                     "Shift $\\Delta$", "Density $g(\\hat q)$", "Predicted coverage", "Observed coverage",
                     "Prediction error (pp)"],
        "Value": [df["R_cal_n"].iloc[0], df["R_test_n"].iloc[0], df["R_cal_mean"].iloc[0],
                  df["R_test_mean"].iloc[0], df["delta"].iloc[0], df["density_at_qhat"].iloc[0],
                  df["predicted_coverage"].iloc[0], df["observed_coverage"].iloc[0],
                  df["prediction_error_pp"].iloc[0]],
    })
    save_table(out, OUT / "table4_theory_validation", index=False,
               caption="First-order drift-sensitivity prediction (Proposition 2) validated against "
                       "the medical no-show temporal split: predicted coverage is computed from "
                       "calibration-period statistics alone.",
               label="tab:theory-validation",
               column_format="lr")
    print("Saved Table 4")


if __name__ == "__main__":
    table1_simulation_sweep()
    table2_real_data_comparison()
    table3_decision_layer_real_data()
    table4_theory_validation()
    print(f"\nAll main tables saved to {OUT}")
