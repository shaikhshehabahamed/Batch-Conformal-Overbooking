"""Final, consolidated real-data validation on the medical no-show dataset:
EDA (batch heterogeneity), the random-split best case, the temporal-split
exchangeability stress test (static calibration), the ACI gamma sweep, and
the final method comparison under both splits. Supersedes the exploratory
exp00/exp04/exp05/exp06/exp07/exp08 scripts -- one script, saved structured
output, nothing redundant.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.clean_noshow import load_and_clean, build_batches, FEATURE_COLS
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
    each method, from the SAME scorer closure (methods.make_method_scorers)
    so the point estimate and its CI can't silently drift apart -- see
    stats_utils.bootstrap_coverage_width's docstring for why batches
    (not patients) are the resampling unit and why the frozen classifier
    is not refit per replicate.
    """
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

    # --- EDA: batch heterogeneity not captured by patient-level covariates ---
    print("=== EDA: attendance rate by top-10 neighbourhoods (by volume) ===")
    top_n = df["Neighbourhood"].value_counts().head(10).index
    by_nbhd = df[df["Neighbourhood"].isin(top_n)].groupby("Neighbourhood")["Attended"].agg(["mean", "count"])
    print(by_nbhd.sort_values("mean"))
    by_nbhd.to_csv(OUT_DIR / "noshow_eda_neighbourhood.csv")

    print("\n=== EDA: attendance rate by day-of-week ===")
    by_dow = df.groupby("day_of_week")["Attended"].agg(["mean", "count"])
    print(by_dow.sort_values("mean"))

    # --- RANDOM split: best case ---
    rng = np.random.default_rng(42)
    shuffled = list(all_batches)
    rng.shuffle(shuffled)
    n_b = len(shuffled)
    train_b, cal_b, test_b = shuffled[:int(0.4*n_b)], shuffled[int(0.4*n_b):int(0.7*n_b)], shuffled[int(0.7*n_b):]
    train_patient_ids = set(np.concatenate([b["patient_ids"] for b in train_b]))
    train_df = df[df["PatientId"].isin(train_patient_ids)]
    clf_random = LogisticRegression(max_iter=500)
    clf_random.fit(train_df[FEATURE_COLS].to_numpy(dtype=float), train_df["Attended"].to_numpy(dtype=int))
    print("\n=== RANDOM split (best case) ===")
    for method, r in static_comparison(cal_b, test_b, clf_random, boot_seed=100).items():
        print(f"  {method:14s} coverage={r['coverage']:.3f} [{r['coverage_ci_lo']:.3f},{r['coverage_ci_hi']:.3f}]"
              f"  width={r['width']:.2f} [{r['width_ci_lo']:.2f},{r['width_ci_hi']:.2f}]")
        rows.append({"split": "random", "method": method, **r})

    # --- TEMPORAL split: static calibration (exchangeability stress test) ---
    dates = sorted(df["AppointmentDay"].dt.date.unique())
    n_d = len(dates)
    train_dates, cal_dates, test_dates = dates[:int(0.4*n_d)], dates[int(0.4*n_d):int(0.7*n_d)], dates[int(0.7*n_d):]
    train_df_t = df[df["AppointmentDay"].dt.date.isin(train_dates)]
    clf_temporal = LogisticRegression(max_iter=500)
    clf_temporal.fit(train_df_t[FEATURE_COLS].to_numpy(dtype=float), train_df_t["Attended"].to_numpy(dtype=int))
    by_date = {}
    for b in all_batches:
        by_date.setdefault(b["appointment_day"].date(), []).append(b)
    cal_temporal = [b for d in cal_dates for b in by_date.get(d, [])]
    test_temporal = [b for d in test_dates for b in by_date.get(d, [])]
    print("\n=== TEMPORAL split, static calibration (exchangeability stress test) ===")
    for method, r in static_comparison(cal_temporal, test_temporal, clf_temporal, boot_seed=200).items():
        print(f"  {method:14s} coverage={r['coverage']:.3f} [{r['coverage_ci_lo']:.3f},{r['coverage_ci_hi']:.3f}]"
              f"  width={r['width']:.2f} [{r['width_ci_lo']:.2f},{r['width_ci_hi']:.2f}]")
        rows.append({"split": "temporal_static", "method": method, **r})

    # --- drift quantification (feeds theory validation script) ---
    cal_attendance = df[df["AppointmentDay"].dt.date.isin(cal_dates)]["Attended"].mean()
    test_attendance = df[df["AppointmentDay"].dt.date.isin(test_dates)]["Attended"].mean()
    print(f"\ncal window attendance rate: {cal_attendance:.4f}")
    print(f"test window attendance rate: {test_attendance:.4f}")
    print(f"shift: {test_attendance - cal_attendance:.4f}")

    # --- ACI gamma sweep ---
    print("\n=== TEMPORAL split, ACI gamma sweep ===")
    aci_rows = []
    for gamma in [0.02, 0.05, 0.10, 0.20]:
        seed_dates = cal_dates[:3]
        adapt_dates = cal_dates[3:] + test_dates
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
        print(f"  gamma={gamma}: coverage={cov:.3f}  width={width:.2f}  final_alpha_t={alpha_traj[-1]:.4f}")
        aci_rows.append({"gamma": gamma, "coverage": cov, "width": width, "final_alpha_t": alpha_traj[-1]})
        if gamma == 0.02:
            # block bootstrap (not the plain batch-resampling bootstrap):
            # ACI's alpha_t and running score pool carry over day-to-day, so
            # only a contiguous-block resample of the day sequence preserves
            # that short-range dependence -- see stats_utils.block_bootstrap_
            # aci_ci's docstring.
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

    pd.DataFrame(rows).to_csv(OUT_DIR / "noshow_final_comparison.csv", index=False)
    pd.DataFrame(aci_rows).to_csv(OUT_DIR / "noshow_aci_gamma_sweep.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'noshow_final_comparison.csv'} and noshow_aci_gamma_sweep.csv")
