"""Final, consolidated downstream decision-layer evaluation on the medical
no-show dataset: translates each method's calibrated bound into an actual
overbooking policy and measures realized cost/overflow/utilization, swept
across three capacity fractions (70/85/95% of historical demand), for the
static baselines AND our ACI-corrected method. Supersedes exp09/10/11/14.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.clean_noshow import load_and_clean, build_batches, FEATURE_COLS
from batch_risk_control.methods import (
    _mu_sigma, naive_parametric_bound, dro_cantelli_bound,
    fit_dro_cantelli_calibrated, fit_bonferroni_individual, bonferroni_bound_single,
)
from batch_risk_control.decision import max_bookable, evaluate_policy, summarize_policy

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10
GAMMA = 0.02
COST_OVERFLOW_PER_PATIENT = 3.0
COST_IDLE_PER_SLOT = 1.0
CAPACITY_FRACTIONS = [0.70, 0.85, 0.95]


def predicted_p_real(batch, clf):
    return clf.predict_proba(batch["X"])[:, 1]


if __name__ == "__main__":
    df = load_and_clean()
    all_batches = build_batches(df)
    dates = sorted(df["AppointmentDay"].dt.date.unique())
    n_d = len(dates)
    train_dates = dates[: int(0.4 * n_d)]
    cal_dates = dates[int(0.4 * n_d): int(0.7 * n_d)]
    test_dates = dates[int(0.7 * n_d):]

    train_df = df[df["AppointmentDay"].dt.date.isin(train_dates)]
    clf = LogisticRegression(max_iter=500)
    clf.fit(train_df[FEATURE_COLS].to_numpy(dtype=float), train_df["Attended"].to_numpy(dtype=int))

    by_date = {}
    for b in all_batches:
        by_date.setdefault(b["appointment_day"].date(), []).append(b)
    cal_batches = [b for d in cal_dates for b in by_date.get(d, [])]
    test_batches = [b for d in test_dates for b in by_date.get(d, []) if len(b["Y"]) >= 5]
    cal_p_hat = [predicted_p_real(b, clf) for b in cal_batches]
    test_p_hat = [predicted_p_real(b, clf) for b in test_batches]

    dro_fit = fit_dro_cantelli_calibrated(cal_batches, cal_p_hat, alpha=ALPHA)
    bonf_fit = fit_bonferroni_individual(np.concatenate(cal_p_hat), np.concatenate([b["Y"] for b in cal_batches]),
                                          n_bins=8)

    def naive_fn(p): return naive_parametric_bound([{"Y": None}], [p], alpha=ALPHA)[0]
    def dro_plugin_fn(p): return dro_cantelli_bound([{"Y": None}], [p], alpha=ALPHA)[0]
    def dro_calib_fn(p):
        mu = np.sum(p); k = np.sqrt((1 - ALPHA) / ALPHA)
        return mu + k * dro_fit["sigma_hist"]
    def bonferroni_fn(p): return bonferroni_bound_single(bonf_fit, p, alpha=ALPHA)

    # ACI q_hat trajectory (walk-forward, classifier frozen), reused across all capacity fractions
    seed_dates = cal_dates[:3]
    adapt_dates = cal_dates[3:] + test_dates
    seed_batches = [b for d in seed_dates for b in by_date.get(d, [])]
    R_pool = []
    for b in seed_batches:
        p_hat = predicted_p_real(b, clf)
        mu, sigma = _mu_sigma(p_hat)
        R_pool.append((np.sum(b["Y"]) - mu) / sigma)
    alpha_t = ALPHA
    q_hat_by_batch_key = {}
    for d in adapt_dates:
        day_batches = [b for b in by_date.get(d, []) if len(b["Y"]) >= 5]
        if not day_batches:
            continue
        a_eff = min(max(alpha_t, 1e-3), 0.999)
        n_pool = len(R_pool)
        k = int(np.ceil((n_pool + 1) * (1 - a_eff)))
        k = min(max(k, 1), n_pool)
        q_hat = np.sort(R_pool)[k - 1]
        for b in day_batches:
            p_hat = predicted_p_real(b, clf)
            q_hat_by_batch_key[b["batch_key"]] = q_hat
            mu_full, sigma_full = _mu_sigma(p_hat)
            S_full = np.sum(b["Y"])
            err = 1.0 if S_full > (mu_full + q_hat * sigma_full) else 0.0
            alpha_t = alpha_t + GAMMA * (ALPHA - err)
            R_pool.append((S_full - mu_full) / sigma_full)

    rows = []
    for frac in CAPACITY_FRACTIONS:
        capacities_all = {b["batch_key"]: max(1, round(frac * len(b["Y"]))) for b in test_batches}
        caps_list = [capacities_all[b["batch_key"]] for b in test_batches]
        print(f"--- capacity = {frac:.0%} of historical demand ---")

        for name, fn in [("naive", naive_fn), ("dro_plugin", dro_plugin_fn),
                          ("dro_calib", dro_calib_fn), ("bonferroni", bonferroni_fn)]:
            records = evaluate_policy(test_batches, test_p_hat, fn, caps_list)
            s = summarize_policy(records)
            cost = COST_OVERFLOW_PER_PATIENT * s["mean_overflow_amount"] + COST_IDLE_PER_SLOT * s["mean_idle_slots"]
            print(f"  {name:12s} overflow_rate={s['overflow_rate']:.3f}  utilization={s['mean_utilization']:.3f}  cost={cost:.2f}")
            rows.append({"capacity_fraction": frac, "method": name, "cost": cost, **s})

        aci_records = []
        for b, p_hat in zip(test_batches, test_p_hat):
            q_hat = q_hat_by_batch_key[b["batch_key"]]
            cap = capacities_all[b["batch_key"]]

            def bound_fn(p, qh=q_hat):
                mu, sigma = _mu_sigma(p)
                return mu + qh * sigma

            n_booked = max_bookable(p_hat, bound_fn, cap)
            realized = int(np.sum(b["Y"][:n_booked])) if n_booked > 0 else 0
            aci_records.append({"n_booked": n_booked, "realized": realized, "capacity": cap,
                                 "overflow": realized > cap, "utilization": min(realized, cap) / cap,
                                 "idle_slots": max(0, cap - realized)})
        s = summarize_policy(aci_records)
        cost = COST_OVERFLOW_PER_PATIENT * s["mean_overflow_amount"] + COST_IDLE_PER_SLOT * s["mean_idle_slots"]
        print(f"  {'ours_ACI':12s} overflow_rate={s['overflow_rate']:.3f}  utilization={s['mean_utilization']:.3f}  cost={cost:.2f}\n")
        rows.append({"capacity_fraction": frac, "method": "ours_ACI", "cost": cost, **s})

    pd.DataFrame(rows).to_csv(OUT_DIR / "decision_layer_real_data.csv", index=False)
    print(f"Saved to {OUT_DIR / 'decision_layer_real_data.csv'}")
