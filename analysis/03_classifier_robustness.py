"""Robustness check: does the qualitative finding (static calibration fails
under real temporal drift, ACI restores validity, remains tighter than
alternatives) depend on using logistic regression specifically as the
underlying classifier for p_hat? Repeats the medical no-show temporal-split
comparison (from the real-data validation) with a gradient-boosted
classifier (HistGradientBoostingClassifier) instead.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.clean_noshow import load_and_clean, build_batches, FEATURE_COLS
from batch_risk_control.methods import make_method_scorers
from batch_risk_control.adaptive import run_aci_batch_conformal

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10


def predicted_p_real(batch, clf):
    return clf.predict_proba(batch["X"])[:, 1]


def evaluate_static(cal_batches, test_batches, clf):
    """Point estimate for each method via the shared scorer factory
    (methods.make_method_scorers) -- same implementation the real-data
    Table 2 comparison uses, so this robustness check can't silently drift
    from what "naive"/"dro_plugin"/etc. mean there."""
    cal_p_hat = [predicted_p_real(b, clf) for b in cal_batches]
    test_p_hat = [predicted_p_real(b, clf) for b in test_batches]
    scorers = make_method_scorers(ALPHA)
    results = {}
    for name, scorer in scorers.items():
        out_name = "ours_static" if name == "ours" else name
        results[out_name] = scorer(cal_batches, cal_p_hat, test_batches, test_p_hat)
    return results


def evaluate_aci(cal_dates, test_dates, by_date, clf, gamma=0.02):
    seed_dates = cal_dates[:3]
    adapt_dates = cal_dates[3:] + test_dates
    seed_batches = [b for d in seed_dates for b in by_date.get(d, [])]
    seed_p_hat = [predicted_p_real(b, clf) for b in seed_batches]
    adapt_batches_by_day = [by_date[d] for d in adapt_dates if d in by_date]
    adapt_p_hat_by_day = [[predicted_p_real(b, clf) for b in by_date[d]] for d in adapt_dates if d in by_date]

    bounds, _ = run_aci_batch_conformal(seed_batches, seed_p_hat, adapt_batches_by_day,
                                         adapt_p_hat_by_day, alpha=ALPHA, gamma=gamma)
    S_all = np.concatenate([[np.sum(b["Y"]) for b in day] for day in adapt_batches_by_day])
    is_test = np.concatenate([[d in test_dates] * len(by_date[d]) for d in adapt_dates if d in by_date])
    S_test, bounds_test = S_all[is_test], bounds[is_test]
    return np.mean(S_test <= bounds_test), np.mean(bounds_test)


if __name__ == "__main__":
    df = load_and_clean()
    all_batches = build_batches(df)
    dates = sorted(df["AppointmentDay"].dt.date.unique())
    n_d = len(dates)
    train_dates = dates[: int(0.4 * n_d)]
    cal_dates = dates[int(0.4 * n_d): int(0.7 * n_d)]
    test_dates = dates[int(0.7 * n_d):]

    train_df = df[df["AppointmentDay"].dt.date.isin(train_dates)]
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    Y_train = train_df["Attended"].to_numpy(dtype=int)

    by_date = {}
    for b in all_batches:
        by_date.setdefault(b["appointment_day"].date(), []).append(b)
    cal_batches = [b for d in cal_dates for b in by_date.get(d, [])]
    test_batches = [b for d in test_dates for b in by_date.get(d, [])]

    rows = []
    for clf_name, clf in [
        ("logistic_regression", LogisticRegression(max_iter=500)),
        ("gradient_boosting", HistGradientBoostingClassifier(max_depth=3, random_state=0)),
    ]:
        clf.fit(X_train, Y_train)
        print(f"\n=== Classifier: {clf_name} ===")

        static_results = evaluate_static(cal_batches, test_batches, clf)
        for method, (cov, width) in static_results.items():
            print(f"  {method:14s} coverage={cov:.3f}  width={width:.2f}")
            rows.append({"classifier": clf_name, "method": method, "coverage": cov, "width": width})

        aci_cov, aci_width = evaluate_aci(cal_dates, test_dates, by_date, clf)
        print(f"  {'ours_ACI':14s} coverage={aci_cov:.3f}  width={aci_width:.2f}")
        rows.append({"classifier": clf_name, "method": "ours_ACI", "coverage": aci_cov, "width": aci_width})

    pd.DataFrame(rows).to_csv(OUT_DIR / "classifier_robustness.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'classifier_robustness.csv'}")
