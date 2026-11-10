"""Validates the first-order drift-sensitivity formula (theory.md,
Proposition 2) against real data: predicts the coverage drop under
temporal drift using ONLY calibration-period statistics, then checks
against the actually-observed test-period coverage. Supersedes exp12.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.clean_noshow import load_and_clean, build_batches, FEATURE_COLS
from batch_risk_control.methods import _mu_sigma

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10


def predicted_p_real(batch, clf):
    return clf.predict_proba(batch["X"])[:, 1]


def compute_R(batches, clf):
    R = []
    for b in batches:
        p_hat = predicted_p_real(b, clf)
        mu, sigma = _mu_sigma(p_hat)
        R.append((np.sum(b["Y"]) - mu) / sigma)
    return np.array(R)


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
    test_batches = [b for d in test_dates for b in by_date.get(d, [])]

    R_cal = compute_R(cal_batches, clf)
    R_test = compute_R(test_batches, clf)

    print(f"R_cal:  n={len(R_cal)}  mean={R_cal.mean():.4f}  std={R_cal.std():.4f}")
    print(f"R_test: n={len(R_test)}  mean={R_test.mean():.4f}  std={R_test.std():.4f}")
    delta = R_test.mean() - R_cal.mean()
    print(f"shift Delta = mean(R_test) - mean(R_cal) = {delta:.4f}")
    print(f"Delta in units of std(R_cal): {delta / R_cal.std():.3f}; of std(R_test): {delta / R_test.std():.3f}")

    n = len(R_cal)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    k = min(k, n)
    q_hat = np.sort(R_cal)[k - 1]
    print(f"q_hat (calibrated on R_cal, alpha={ALPHA}) = {q_hat:.4f}")

    kde = gaussian_kde(R_cal)
    g_at_qhat = kde(q_hat)[0]
    print(f"estimated density g(q_hat) from R_cal (KDE) = {g_at_qhat:.4f}")

    predicted_coverage = (1 - ALPHA) - delta * g_at_qhat
    observed_coverage = np.mean(R_test <= q_hat)
    print(f"\nfirst-order predicted coverage = (1-alpha) - Delta*g(q_hat) = {predicted_coverage:.4f}")
    print(f"actual observed coverage P(R_test <= q_hat) = {observed_coverage:.4f}")
    print(f"prediction error: {abs(predicted_coverage - observed_coverage)*100:.2f} percentage points")

    pd.DataFrame([{
        "R_cal_n": len(R_cal), "R_cal_mean": R_cal.mean(), "R_cal_std": R_cal.std(),
        "R_test_n": len(R_test), "R_test_mean": R_test.mean(), "R_test_std": R_test.std(),
        "delta": delta, "q_hat": q_hat, "density_at_qhat": g_at_qhat,
        "predicted_coverage": predicted_coverage, "observed_coverage": observed_coverage,
        "prediction_error_pp": abs(predicted_coverage - observed_coverage) * 100,
    }]).to_csv(OUT_DIR / "theory_drift_validation.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'theory_drift_validation.csv'}")
