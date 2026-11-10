"""Generates supplementary paper tables from saved analysis outputs."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from batch_risk_control.tables_utils import save_table
from batch_risk_control.plotting import METHOD_LABELS

DATA = ROOT / "data" / "processed"
OUT = ROOT / "paper_assets" / "tables" / "supplementary"


def tableS1_width_significance_tests():
    df = pd.read_csv(DATA / "simulation_study_tests.csv")
    # exact-match, not substring: "ours vs nominal 0.90" also contains "ours vs"
    # and is a one-sample coverage test, not a paired width comparison -- it
    # does not belong in this table and its ci95/mean_diff mean something
    # different (coverage mean and its CI, not a width difference).
    df = df[df["label"].isin(["ours vs bonferroni", "ours vs dro_calib", "ours vs naive"])]
    out = df[["tau", "label", "mean_diff", "ci95_lo", "ci95_hi", "wilcoxon_p"]].copy()
    out.columns = ["$\\tau$", "Comparison", "Mean diff. (width)", "95\\% CI lo", "95\\% CI hi", "Wilcoxon $p$"]
    save_table(out, OUT / "tableS1_width_significance_tests", index=False,
               caption="Paired Wilcoxon significance tests: our method's width vs. Bonferroni "
                       "and DRO-calibrated, across 50 matched replications per $\\tau$, plus a "
                       "two-sided check against naive parametric at $\\tau=0$ (the one point "
                       "where naive is itself still valid).",
               label="tab:width-tests", column_format="llrrrr")
    print("Saved Table S1")


def tableS2_alpha_sensitivity():
    df = pd.read_csv(DATA / "alpha_sensitivity_tests.csv")
    out = df[["alpha", "target", "mean_coverage", "ci95_lo", "ci95_hi", "ttest_p"]].copy()
    out.columns = ["$\\alpha$", "Target ($1-\\alpha$)", "Mean coverage", "95\\% CI lo", "95\\% CI hi", "$p$ (vs. target)"]
    save_table(out, OUT / "tableS2_alpha_sensitivity", index=False,
               caption="Coverage vs. nominal target across $\\alpha \\in \\{0.05, 0.10, 0.20\\}$ "
                       "at $\\tau=1.2$ (50 replications each). No $p$-value is significant at 0.05.",
               label="tab:alpha-sensitivity", column_format="lrrrrr")
    print("Saved Table S2")


def tableS3_ablation():
    df = pd.read_csv(DATA / "calibration_ablation.csv")
    out = df[["method", "coverage", "mean_width"]].copy()
    out.columns = ["Calibration approach", "Coverage", "Mean width"]
    save_table(out, OUT / "tableS3_calibration_ablation", index=False,
               caption="Calibration ablation on the medical no-show temporal split: incremental "
                       "fixes vs. the adopted ACI fix.", label="tab:ablation", column_format="lrr")
    print("Saved Table S3")


def tableS4_classifier_robustness():
    df = pd.read_csv(DATA / "classifier_robustness.csv")
    df["Method"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    out = df[["classifier", "Method", "coverage", "width"]].copy()
    out.columns = ["Classifier", "Method", "Coverage", "Width"]
    out["Classifier"] = out["Classifier"].str.replace("_", " ").str.title()
    save_table(out, OUT / "tableS4_classifier_robustness", index=False,
               caption="Classifier robustness: logistic regression vs. gradient boosting, "
                       "medical no-show temporal split.", label="tab:classifier-robustness",
               column_format="llrr")
    print("Saved Table S4")


def tableS5_aci_gamma_sweep():
    df = pd.read_csv(DATA / "noshow_aci_gamma_sweep.csv")
    out = df[["gamma", "coverage", "width", "final_alpha_t"]].copy()
    out.columns = ["$\\gamma$", "Coverage", "Width", "Final $\\alpha_t$"]
    save_table(out, OUT / "tableS5_aci_gamma_sweep", index=False,
               caption="ACI step-size ($\\gamma$) sweep on the medical no-show temporal split.",
               label="tab:aci-gamma", column_format="lrrr")
    print("Saved Table S5")


def tableS6_decision_layer_simulation():
    df = pd.read_csv(DATA / "decision_layer_simulation_raw.csv")
    agg = df.groupby(["tau", "method"])[["overflow_rate", "utilization", "cost"]].mean().reset_index()
    agg["Method"] = agg["method"].map(METHOD_LABELS).fillna(agg["method"])
    out = agg[["tau", "Method", "overflow_rate", "utilization", "cost"]].copy()
    out.columns = ["$\\tau$", "Method", "Overflow rate", "Utilization", "Cost"]
    save_table(out, OUT / "tableS6_decision_layer_simulation", index=False,
               caption="Downstream decision-layer outcomes on simulated data across the "
                       "correlation sweep (30 replications per point, capacity = 85\\% of demand).",
               label="tab:decision-simulation", column_format="llrrr")
    print("Saved Table S6")


def tableS7_decision_layer_stats():
    df = pd.read_csv(DATA / "decision_layer_simulation_tests.csv")
    out = df[["tau", "comparison", "mean_diff", "ci95_lo", "ci95_hi"]].copy()
    # the three "vs 10% target" rows are one-sample t-tests with no Wilcoxon
    # stat (wilcoxon_p is NaN there) -- fall back to ttest_p so the table
    # shows the real p-value RESULTS.md cites instead of a literal "nan".
    out["p"] = df["wilcoxon_p"].fillna(df["ttest_p"])
    out.columns = ["$\\tau$", "Comparison", "Mean diff.", "95\\% CI lo", "95\\% CI hi", "$p$"]
    save_table(out, OUT / "tableS7_decision_layer_stats", index=False,
               caption="Significance tests for the simulated decision-layer results: paired "
                       "Wilcoxon tests (overflow rate and cost comparisons) and one-sample "
                       "$t$-tests (overflow rate vs.\\ the 10\\% target, $\\tau=2.0$).",
               label="tab:decision-stats", column_format="llrrrr")
    print("Saved Table S7")


if __name__ == "__main__":
    tableS1_width_significance_tests()
    tableS2_alpha_sensitivity()
    tableS3_ablation()
    tableS4_classifier_robustness()
    tableS5_aci_gamma_sweep()
    tableS6_decision_layer_simulation()
    tableS7_decision_layer_stats()
    print(f"\nAll supplementary tables saved to {OUT}")
