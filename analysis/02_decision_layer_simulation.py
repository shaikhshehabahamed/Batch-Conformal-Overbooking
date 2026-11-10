"""Applies the overbooking decision evaluation (decision.py) to SIMULATED
batches across the same within-batch correlation (tau) sweep used in the
interval-level validation. This directly tests whether DRO-plugin's real-
data cost advantage (found in the real no-show decision analysis) is a
general property or specific to that dataset's particular (moderate) drift
magnitude: DRO-plugin's fixed Cantelli margin is proven to lose validity at
high tau (interval-level: 73.0% coverage at tau=2.0, well below the 90%
target) -- if that invalidity also shows up as an unfair cost advantage in
the decision layer (booking too aggressively because its own bound is too
narrow to be trustworthy), that directly explains why the real-data finding
should not be read as a general recommendation for DRO-plugin.

No ACI here: the simulation's tau controls WITHIN-batch correlation, not
BETWEEN-batch temporal drift, so there is nothing for ACI to adapt to in
this controlled setting -- this script isolates the correlation-robustness
question cleanly, complementing (not duplicating) the real-data drift
analysis.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.dgp import generate_batches, predicted_p
from batch_risk_control.methods import (
    naive_parametric_bound, fit_batch_conformal,
    fit_bonferroni_individual, bonferroni_bound_single,
    dro_cantelli_bound, fit_dro_cantelli_calibrated,
)
from batch_risk_control.decision import evaluate_policy, summarize_policy

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
N_FEATURES = 3
MAX_BATCH_SIZE = 80
N_TRAIN_BATCHES = 150
N_CAL = 250
N_TEST = 200
N_REPS = 30
CAPACITY_FRACTION = 0.85
COST_OVERFLOW_PER_PATIENT = 3.0
COST_IDLE_PER_SLOT = 1.0


def run_one_replication(tau, seed):
    rng = np.random.default_rng(seed)
    all_batches = generate_batches(n_batches=N_TRAIN_BATCHES + N_CAL + N_TEST,
                                    batch_size_range=(20, MAX_BATCH_SIZE),
                                    n_features=N_FEATURES, tau=tau, rng=rng)
    train_batches = all_batches[:N_TRAIN_BATCHES]
    X_train = np.vstack([b["X"] for b in train_batches])
    Y_train = np.concatenate([b["Y"] for b in train_batches])
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    beta_hat = clf.coef_.ravel()

    remaining = all_batches[N_TRAIN_BATCHES:]
    cal_batches = remaining[:N_CAL]
    test_batches = remaining[N_CAL:N_CAL + N_TEST]
    cal_p_hat = [predicted_p(b, beta_hat) for b in cal_batches]
    test_p_hat = [predicted_p(b, beta_hat) for b in test_batches]
    capacities = [max(1, round(CAPACITY_FRACTION * len(b["Y"]))) for b in test_batches]

    dro_fit = fit_dro_cantelli_calibrated(cal_batches, cal_p_hat, alpha=ALPHA)
    conf_fit = fit_batch_conformal(cal_batches, cal_p_hat, alpha=ALPHA)
    cal_patients_p = np.concatenate(cal_p_hat)
    cal_patients_y = np.concatenate([b["Y"] for b in cal_batches])
    bonf_fit = fit_bonferroni_individual(cal_patients_p, cal_patients_y, n_bins=8)

    def naive_fn(p): return naive_parametric_bound([{"Y": None}], [p], alpha=ALPHA)[0]
    def dro_plugin_fn(p): return dro_cantelli_bound([{"Y": None}], [p], alpha=ALPHA)[0]
    def dro_calib_fn(p):
        mu = np.sum(p); k = np.sqrt((1 - ALPHA) / ALPHA)
        return mu + k * dro_fit["sigma_hist"]
    def ours_fn(p):
        mu = np.sum(p); sigma = np.sqrt(max(np.sum(p * (1 - p)), 1e-8))
        return mu + conf_fit["q_hat"] * sigma
    def bonferroni_fn(p): return bonferroni_bound_single(bonf_fit, p, alpha=ALPHA)

    out = {}
    for name, fn in [("naive", naive_fn), ("dro_plugin", dro_plugin_fn), ("dro_calib", dro_calib_fn),
                      ("ours", ours_fn), ("bonferroni", bonferroni_fn)]:
        records = evaluate_policy(test_batches, test_p_hat, fn, capacities)
        s = summarize_policy(records)
        cost = COST_OVERFLOW_PER_PATIENT * s["mean_overflow_amount"] + COST_IDLE_PER_SLOT * s["mean_idle_slots"]
        out[name] = {"overflow_rate": s["overflow_rate"], "utilization": s["mean_utilization"],
                     "cost": cost, "mean_n_booked": s["mean_n_booked"]}
    return out


if __name__ == "__main__":
    rows = []
    t0 = time.time()
    for tau in [0.0, 0.5, 1.2, 2.0]:
        for rep in range(N_REPS):
            result = run_one_replication(tau, seed=30_000 + rep)
            for method, metrics in result.items():
                rows.append({"tau": tau, "replication": rep, "method": method, **metrics})
        print(f"  tau={tau} done, elapsed={time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "decision_layer_simulation_raw.csv", index=False)
    print(f"\nSaved {len(df)} rows to {OUT_DIR / 'decision_layer_simulation_raw.csv'}")

    print("\n=== Mean cost and overflow rate by method x tau ===")
    summary = df.groupby(["tau", "method"])[["overflow_rate", "utilization", "cost"]].mean().round(3)
    print(summary)
