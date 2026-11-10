"""Ablation: is ACI's fix earning its complexity, or would a simpler fix
for temporal drift have been just as good? Compares, on the same medical
no-show temporal split: (1) static calibration (no adaptation), (2) rolling
calibration only (10-day window, classifier frozen), (3) rolling classifier
AND calibration (both retrained on the same window), (4) ACI (gamma=0.02).
Consolidates what were exploratory diagnostic scripts into a single
reproducible ablation table.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.clean_noshow import load_and_clean, build_batches, FEATURE_COLS
from batch_risk_control.methods import fit_batch_conformal, batch_conformal_bound
from batch_risk_control.adaptive import run_aci_batch_conformal

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10
ROLLING_WINDOW_DAYS = 10


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
    by_date = {}
    for b in all_batches:
        by_date.setdefault(b["appointment_day"].date(), []).append(b)

    rows = []

    # (1) Static: classifier + calibration both frozen from the initial period
    train_df = df[df["AppointmentDay"].dt.date.isin(train_dates)]
    clf_static = LogisticRegression(max_iter=500)
    clf_static.fit(train_df[FEATURE_COLS].to_numpy(dtype=float), train_df["Attended"].to_numpy(dtype=int))
    cal_batches = [b for d in cal_dates for b in by_date.get(d, [])]
    test_batches = [b for d in test_dates for b in by_date.get(d, [])]
    cal_p = [predicted_p_real(b, clf_static) for b in cal_batches]
    test_p = [predicted_p_real(b, clf_static) for b in test_batches]
    fit = fit_batch_conformal(cal_batches, cal_p, alpha=ALPHA)
    bounds = batch_conformal_bound(fit, test_batches, test_p)
    S_test = np.array([np.sum(b["Y"]) for b in test_batches])
    rows.append({"method": "static (no adaptation)", "coverage": np.mean(S_test <= bounds), "mean_width": np.mean(bounds)})

    # (2) Rolling calibration only (classifier still frozen from static fit)
    S_list, bounds_list = [], []
    for td in test_dates:
        idx = dates.index(td)
        window_dates = dates[max(0, idx - ROLLING_WINDOW_DAYS): idx]
        wcal = [b for d in window_dates for b in by_date.get(d, [])]
        wtest = by_date.get(td, [])
        if len(wcal) < 20 or not wtest:
            continue
        wcal_p = [predicted_p_real(b, clf_static) for b in wcal]
        wtest_p = [predicted_p_real(b, clf_static) for b in wtest]
        f = fit_batch_conformal(wcal, wcal_p, alpha=ALPHA)
        b_ = batch_conformal_bound(f, wtest, wtest_p)
        S_list.append([np.sum(b["Y"]) for b in wtest])
        bounds_list.append(b_)
    S_roll = np.concatenate(S_list)
    bounds_roll = np.concatenate(bounds_list)
    rows.append({"method": "rolling calibration only", "coverage": np.mean(S_roll <= bounds_roll), "mean_width": np.mean(bounds_roll)})

    # (3) Rolling classifier AND calibration (both retrained each window)
    S_list, bounds_list = [], []
    for td in test_dates:
        idx = dates.index(td)
        window_dates = dates[max(0, idx - ROLLING_WINDOW_DAYS): idx]
        wcal = [b for d in window_dates for b in by_date.get(d, [])]
        wtest = by_date.get(td, [])
        if len(wcal) < 20 or not wtest:
            continue
        window_df = df[df["AppointmentDay"].dt.date.isin(window_dates)]
        clf_roll = LogisticRegression(max_iter=500)
        clf_roll.fit(window_df[FEATURE_COLS].to_numpy(dtype=float), window_df["Attended"].to_numpy(dtype=int))
        wcal_p = [predicted_p_real(b, clf_roll) for b in wcal]
        wtest_p = [predicted_p_real(b, clf_roll) for b in wtest]
        f = fit_batch_conformal(wcal, wcal_p, alpha=ALPHA)
        b_ = batch_conformal_bound(f, wtest, wtest_p)
        S_list.append([np.sum(b["Y"]) for b in wtest])
        bounds_list.append(b_)
    S_roll2 = np.concatenate(S_list)
    bounds_roll2 = np.concatenate(bounds_list)
    rows.append({"method": "rolling classifier + calibration", "coverage": np.mean(S_roll2 <= bounds_roll2), "mean_width": np.mean(bounds_roll2)})

    # (4) ACI (gamma=0.02), classifier frozen -- the adopted fix
    seed_dates = cal_dates[:3]
    adapt_dates = cal_dates[3:] + test_dates
    seed_batches = [b for d in seed_dates for b in by_date.get(d, [])]
    seed_p_hat = [predicted_p_real(b, clf_static) for b in seed_batches]
    adapt_batches_by_day = [by_date[d] for d in adapt_dates if d in by_date]
    adapt_p_hat_by_day = [[predicted_p_real(b, clf_static) for b in by_date[d]] for d in adapt_dates if d in by_date]
    bounds_aci, _ = run_aci_batch_conformal(seed_batches, seed_p_hat, adapt_batches_by_day,
                                             adapt_p_hat_by_day, alpha=ALPHA, gamma=0.02)
    S_all = np.concatenate([[np.sum(b["Y"]) for b in day] for day in adapt_batches_by_day])
    is_test = np.concatenate([[d in test_dates] * len(by_date[d]) for d in adapt_dates if d in by_date])
    S_test_aci, bounds_test_aci = S_all[is_test], bounds_aci[is_test]
    rows.append({"method": "ACI (gamma=0.02, adopted fix)", "coverage": np.mean(S_test_aci <= bounds_test_aci),
                 "mean_width": np.mean(bounds_test_aci)})

    out = pd.DataFrame(rows)
    print("=== Calibration ablation, medical no-show temporal split (target coverage 90%) ===")
    print(out.to_string(index=False))
    out.to_csv(OUT_DIR / "calibration_ablation.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'calibration_ablation.csv'}")
