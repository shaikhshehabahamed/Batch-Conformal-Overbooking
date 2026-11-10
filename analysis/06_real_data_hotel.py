"""Final, consolidated second-domain validation on the Hotel Booking Demand
dataset: EDA (seasonality), random split (best case), temporal split
(static, then ACI). Supersedes exp13.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.clean_hotel import load_and_clean, build_batches, FEATURE_COLS
from batch_risk_control.methods import make_method_scorers
from batch_risk_control.stats_utils import bootstrap_coverage_width, block_bootstrap_aci_ci
from batch_risk_control.adaptive import run_aci_batch_conformal

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10
BOOT_B = 1000       # static (batch-resampling) bootstrap replicates
ACI_BOOT_B = 500    # block bootstrap replicates (more expensive: reruns the ACI recursion each time)


def predicted_p_real(batch, clf):
    return clf.predict_proba(batch["X"])[:, 1]


def static_comparison(cal_batches, test_batches, clf, boot_seed=0):
    """Point estimate AND a two-stage batch-level bootstrap 95% CI for
    each method -- see 05_real_data_noshow.py's copy of this function for
    the full rationale (identical here, kept in sync by hand since this
    dataset's build_batches/predicted_p_real differ slightly from the
    no-show dataset's)."""
    cal_p_hat = [predicted_p_real(b, clf) for b in cal_batches]
    test_p_hat = [predicted_p_real(b, clf) for b in test_batches]
    scorers = make_method_scorers(ALPHA)

    results = {}
    for i, (name, scorer) in enumerate(scorers.items()):
        cov, width = scorer(cal_batches, cal_p_hat, test_batches, test_p_hat)
        boot = bootstrap_coverage_width(cal_batches, cal_p_hat, test_batches, test_p_hat,
                                         scorer, B=BOOT_B, seed=boot_seed + i)
        out_name = "ours_static" if name == "ours" else name
        results[out_name] = {
            "coverage": cov, "width": width,
            "coverage_ci_lo": boot["coverage_ci"][0], "coverage_ci_hi": boot["coverage_ci"][1],
            "width_ci_lo": boot["width_ci"][0], "width_ci_hi": boot["width_ci"][1],
        }
    return results


if __name__ == "__main__":
    df = load_and_clean()
    all_batches = build_batches(df)
    rows = []

    print("=== EDA: fulfillment rate by year-month (seasonality) ===")
    by_month = df.groupby(df["arrival_date"].dt.to_period("M"))["Fulfilled"].agg(["mean", "count"])
    print(by_month)
    by_month.to_csv(OUT_DIR / "hotel_eda_monthly.csv")

    # --- RANDOM split ---
    rng = np.random.default_rng(42)
    shuffled = list(all_batches)
    rng.shuffle(shuffled)
    n_b = len(shuffled)
    train_b, cal_b, test_b = shuffled[:int(0.4*n_b)], shuffled[int(0.4*n_b):int(0.7*n_b)], shuffled[int(0.7*n_b):]
    train_batch_keys = set(b["batch_key"] for b in train_b)
    train_df = df[df["batch_key"].isin(train_batch_keys)]
    clf_random = LogisticRegression(max_iter=1000)
    clf_random.fit(train_df[FEATURE_COLS].to_numpy(dtype=float), train_df["Fulfilled"].to_numpy(dtype=int))
    print("\n=== RANDOM split (best case) ===")
    for method, r in static_comparison(cal_b, test_b, clf_random, boot_seed=100).items():
        print(f"  {method:14s} coverage={r['coverage']:.3f} [{r['coverage_ci_lo']:.3f},{r['coverage_ci_hi']:.3f}]"
              f"  width={r['width']:.2f} [{r['width_ci_lo']:.2f},{r['width_ci_hi']:.2f}]")
        rows.append({"split": "random", "method": method, **r})

    # --- TEMPORAL split, static ---
    dates = sorted(df["arrival_date"].dt.date.unique())
    n_d = len(dates)
    train_dates, cal_dates, test_dates = dates[:int(0.4*n_d)], dates[int(0.4*n_d):int(0.7*n_d)], dates[int(0.7*n_d):]
    train_df_t = df[df["arrival_date"].dt.date.isin(train_dates)]
    clf_temporal = LogisticRegression(max_iter=1000)
    clf_temporal.fit(train_df_t[FEATURE_COLS].to_numpy(dtype=float), train_df_t["Fulfilled"].to_numpy(dtype=int))
    by_date = {}
    for b in all_batches:
        by_date.setdefault(b["arrival_date"].date(), []).append(b)
    cal_temporal = [b for d in cal_dates for b in by_date.get(d, [])]
    test_temporal = [b for d in test_dates for b in by_date.get(d, [])]
    print(f"\ntemporal split: train {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d), "
          f"cal ({len(cal_dates)}d), test ({len(test_dates)}d)")
    print("=== TEMPORAL split, static calibration ===")
    for method, r in static_comparison(cal_temporal, test_temporal, clf_temporal, boot_seed=200).items():
        print(f"  {method:14s} coverage={r['coverage']:.3f} [{r['coverage_ci_lo']:.3f},{r['coverage_ci_hi']:.3f}]"
              f"  width={r['width']:.2f} [{r['width_ci_lo']:.2f},{r['width_ci_hi']:.2f}]")
        rows.append({"split": "temporal_static", "method": method, **r})

    # --- TEMPORAL split, ACI ---
    print("\n=== TEMPORAL split, ACI ===")
    for gamma in [0.02, 0.05]:
        seed_dates = cal_dates[:10]
        adapt_dates = cal_dates[10:] + test_dates
        seed_batches = [b for d in seed_dates for b in by_date.get(d, [])]
        seed_p_hat = [predicted_p_real(b, clf_temporal) for b in seed_batches]
        adapt_batches_by_day = [by_date[d] for d in adapt_dates if d in by_date]
        adapt_p_hat_by_day = [[predicted_p_real(b, clf_temporal) for b in by_date[d]] for d in adapt_dates if d in by_date]
        day_is_test = np.array([d in test_dates for d in adapt_dates if d in by_date])
        bounds, alpha_traj = run_aci_batch_conformal(seed_batches, seed_p_hat, adapt_batches_by_day,
                                                      adapt_p_hat_by_day, alpha=ALPHA, gamma=gamma)
        S_all = np.concatenate([[np.sum(b["Y"]) for b in day] for day in adapt_batches_by_day])
        is_test = np.concatenate([[d in test_dates] * len(by_date[d]) for d in adapt_dates if d in by_date])
        S_test, bounds_test = S_all[is_test], bounds[is_test]
        cov, width = np.mean(S_test <= bounds_test), np.mean(bounds_test)
        print(f"  gamma={gamma}: coverage={cov:.3f}  width={width:.2f}  final_alpha_t={alpha_traj[-1]:.3f}")
        if gamma == 0.02:
            # block bootstrap: preserves day-to-day dependence in alpha_t's
            # recursion -- see 05_real_data_noshow.py / stats_utils for the
            # same construction applied to the medical dataset.
            def run_fn(boot_days, boot_p_hat, gamma=gamma):
                return run_aci_batch_conformal(seed_batches, seed_p_hat, boot_days, boot_p_hat,
                                                alpha=ALPHA, gamma=gamma)[0]
            boot = block_bootstrap_aci_ci(adapt_batches_by_day, adapt_p_hat_by_day, day_is_test,
                                           run_fn, B=ACI_BOOT_B, seed=300)
            print(f"    block-bootstrap 95% CI (block_len={boot['block_len']}, B={boot['B']}): "
                  f"coverage [{boot['coverage_ci'][0]:.3f},{boot['coverage_ci'][1]:.3f}]  "
                  f"width [{boot['width_ci'][0]:.2f},{boot['width_ci'][1]:.2f}]")
            rows.append({"split": "temporal_ACI", "method": "ours_ACI", "coverage": cov, "width": width,
                         "coverage_ci_lo": boot["coverage_ci"][0], "coverage_ci_hi": boot["coverage_ci"][1],
                         "width_ci_lo": boot["width_ci"][0], "width_ci_hi": boot["width_ci"][1]})

    pd.DataFrame(rows).to_csv(OUT_DIR / "hotel_final_comparison.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'hotel_final_comparison.csv'}")
