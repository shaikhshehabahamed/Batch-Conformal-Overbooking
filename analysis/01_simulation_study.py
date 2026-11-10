"""Core simulation validation: coverage and width for all five methods
(naive parametric, plug-in DRO/Cantelli, calibrated DRO/Cantelli, our
batch-conformal method, Bonferroni) across a within-batch correlation
sweep (tau), with proper multi-seed replication, confidence intervals, and
paired statistical tests -- not point estimates alone.

Also runs an alpha-level sensitivity sweep at a fixed representative tau,
confirming the qualitative findings aren't an artifact of alpha=0.10.

Output: a single long-format CSV (one row per replication x method x
config) that every downstream figure/table script reads -- the simulation
itself is only ever run here, once.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from batch_risk_control.dgp import generate_batches, predicted_p
from batch_risk_control.methods import make_method_scorers
from batch_risk_control.stats_utils import paired_comparison, coverage_vs_nominal, format_p

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES = 3
MAX_BATCH_SIZE = 80
N_TRAIN_BATCHES = 150
N_CAL = 250
N_TEST = 200
N_REPS = 50  # per config -- generous for tight CIs, per "no need to rush"


def run_one_replication(tau, alpha, seed):
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

    scorers = make_method_scorers(alpha)
    out = {name: scorer(cal_batches, cal_p_hat, test_batches, test_p_hat)
           for name, scorer in scorers.items()}
    return out


def run_sweep(configs, base_seed):
    """configs: list of (config_label, tau, alpha) tuples."""
    rows = []
    t0 = time.time()
    for config_label, tau, alpha in configs:
        for rep in range(N_REPS):
            result = run_one_replication(tau, alpha, seed=base_seed + rep)
            for method, (cov, width) in result.items():
                rows.append({"config": config_label, "tau": tau, "alpha": alpha,
                             "replication": rep, "method": method,
                             "coverage": cov, "width": width})
        elapsed = time.time() - t0
        print(f"  config={config_label} (tau={tau}, alpha={alpha}) done, elapsed={elapsed:.0f}s")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Main tau sweep (alpha=0.10, 50 reps each) ===")
    tau_configs = [(f"tau_{tau}", tau, 0.10) for tau in [0.0, 0.5, 1.2, 2.0]]
    df_tau = run_sweep(tau_configs, base_seed=10_000)

    print("\n=== Alpha sensitivity sweep (tau=1.2, 50 reps each) ===")
    alpha_configs = [(f"alpha_{a}", 1.2, a) for a in [0.05, 0.10, 0.20]]
    df_alpha = run_sweep(alpha_configs, base_seed=20_000)

    df_all = pd.concat([df_tau, df_alpha], ignore_index=True)
    df_all.to_csv(OUT_DIR / "simulation_study_raw.csv", index=False)
    print(f"\nSaved {len(df_all)} rows to {OUT_DIR / 'simulation_study_raw.csv'}")

    # --- Statistical tests on the main tau sweep: is "ours" significantly
    # narrower than Bonferroni / DRO-calibrated, at each tau, among
    # replications where both methods are valid? ---
    print("\n=== Statistical tests: width comparisons (paired across replications) ===")
    test_rows = []
    for tau in [0.0, 0.5, 1.2, 2.0]:
        sub = df_tau[(df_tau["tau"] == tau)]
        ours_w = sub[sub["method"] == "ours"].sort_values("replication")["width"].to_numpy()
        bonf_w = sub[sub["method"] == "bonferroni"].sort_values("replication")["width"].to_numpy()
        droc_w = sub[sub["method"] == "dro_calib"].sort_values("replication")["width"].to_numpy()

        r1 = paired_comparison(ours_w, bonf_w, "ours", "bonferroni", alternative="less")
        r2 = paired_comparison(ours_w, droc_w, "ours", "dro_calib", alternative="less")
        tests_this_tau = [r1, r2]
        if tau == 0.0:
            # at tau=0 there's nothing to correct for -- check the "no
            # tightness penalty when unneeded" claim directly, two-sided
            # (we're not claiming ours is narrower here, just not wider)
            naive_w = sub[sub["method"] == "naive"].sort_values("replication")["width"].to_numpy()
            tests_this_tau.append(paired_comparison(ours_w, naive_w, "ours", "naive", alternative="two-sided"))
        for r in tests_this_tau:
            print(f"  tau={tau}: {r['label']}: mean_diff={r['mean_diff']:.2f} "
                  f"95%CI=({r['ci95'][0]:.2f},{r['ci95'][1]:.2f}) "
                  f"Wilcoxon p={format_p(r['wilcoxon_p'])}")
            test_rows.append({"tau": tau, **{k: v for k, v in r.items() if k != "ci95"},
                               "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1]})

    # --- Coverage vs nominal (90%) tests for "ours" at each tau ---
    print("\n=== Statistical tests: coverage vs. 90% nominal (ours) ===")
    for tau in [0.0, 0.5, 1.2, 2.0]:
        sub = df_tau[(df_tau["tau"] == tau) & (df_tau["method"] == "ours")]
        r = coverage_vs_nominal(sub["coverage"].to_numpy(), nominal=0.90)
        print(f"  tau={tau}: mean={r['mean']:.4f} 95%CI=({r['ci95'][0]:.4f},{r['ci95'][1]:.4f}) "
              f"p={format_p(r['ttest_p'])} (not-significantly-different-from-nominal is the desired result)")
        test_rows.append({"tau": tau, "label": "ours vs nominal 0.90", "n": r["n"],
                           "mean_diff": r["mean"] - 0.90, "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
                           "ttest_stat": r["ttest_stat"], "ttest_p": r["ttest_p"],
                           "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1]})

    pd.DataFrame(test_rows).to_csv(OUT_DIR / "simulation_study_tests.csv", index=False)
    print(f"\nSaved statistical test results to {OUT_DIR / 'simulation_study_tests.csv'}")

    # --- Coverage vs nominal target tests for the alpha sensitivity sweep ---
    print("\n=== Statistical tests: coverage vs. nominal target, alpha sweep (ours, tau=1.2) ===")
    alpha_test_rows = []
    for a in [0.05, 0.10, 0.20]:
        sub = df_alpha[(df_alpha["alpha"] == a) & (df_alpha["method"] == "ours")]
        r = coverage_vs_nominal(sub["coverage"].to_numpy(), nominal=1 - a)
        print(f"  alpha={a}: mean={r['mean']:.4f} target={1-a:.2f} "
              f"95%CI=({r['ci95'][0]:.4f},{r['ci95'][1]:.4f}) p={format_p(r['ttest_p'])}")
        alpha_test_rows.append({"alpha": a, "target": 1 - a, "mean_coverage": r["mean"],
                                 "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1], "ttest_p": r["ttest_p"]})
    pd.DataFrame(alpha_test_rows).to_csv(OUT_DIR / "alpha_sensitivity_tests.csv", index=False)
    print(f"Saved alpha-sensitivity test results to {OUT_DIR / 'alpha_sensitivity_tests.csv'}")
