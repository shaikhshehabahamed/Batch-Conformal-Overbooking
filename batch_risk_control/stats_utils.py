"""Statistical testing utilities shared across analysis scripts: paired
comparisons between methods (width, cost) across Monte Carlo replications,
and one-sample tests of realized coverage against a nominal target.
"""
import numpy as np
from scipy import stats as sp_stats


def paired_comparison(values_a, values_b, label_a="A", label_b="B", alternative="less"):
    """Paired comparison of values_a vs values_b across matched replications
    (e.g. width_ours[r] vs width_bonferroni[r] for replication r). Reports
    both a Wilcoxon signed-rank test (primary; distribution-free, robust to
    non-normal per-replication differences) and a paired t-test (secondary,
    for readers who expect one), plus the mean difference and a normal-
    approximation 95% CI on that difference.

    alternative='less' tests H1: values_a < values_b (e.g. "ours is
    narrower than Bonferroni"), matching scipy's convention.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    assert len(values_a) == len(values_b), "paired comparison requires equal-length matched samples"
    diff = values_a - values_b
    n = len(diff)

    mean_diff = diff.mean()
    se_diff = diff.std(ddof=1) / np.sqrt(n)
    ci_lo, ci_hi = mean_diff - 1.96 * se_diff, mean_diff + 1.96 * se_diff

    if np.allclose(diff, 0):
        wilcoxon_stat, wilcoxon_p = np.nan, 1.0
    else:
        wilcoxon_stat, wilcoxon_p = sp_stats.wilcoxon(values_a, values_b, alternative=alternative)
    t_stat, t_p = sp_stats.ttest_rel(values_a, values_b, alternative=alternative)

    return {
        "label": f"{label_a} vs {label_b}",
        "n": n,
        "mean_diff": mean_diff,
        "ci95": (ci_lo, ci_hi),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
    }


def coverage_vs_nominal(coverage_rates, nominal, alternative="two-sided"):
    """One-sample t-test of replication-level coverage rates against a
    nominal target (e.g. does mean coverage across 30 replications differ
    significantly from 0.90?). Returns the mean, SE, t-statistic, p-value,
    and a 95% CI on the mean.
    """
    coverage_rates = np.asarray(coverage_rates, dtype=float)
    n = len(coverage_rates)
    mean = coverage_rates.mean()
    se = coverage_rates.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    t_stat, p = sp_stats.ttest_1samp(coverage_rates, nominal, alternative=alternative)
    ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
    return {
        "n": n, "mean": mean, "se": se, "ci95": (ci_lo, ci_hi),
        "nominal": nominal, "ttest_stat": t_stat, "ttest_p": p,
    }


def format_p(p):
    """Standard formatting: exact value down to 0.001, '<0.001' below that."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _percentile_ci(samples, ci):
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(np.percentile(samples, lo_pct)), float(np.percentile(samples, hi_pct))


def bootstrap_coverage_width(cal_batches, cal_p_hat, test_batches, test_p_hat,
                              method_fn, B=1000, seed=0, ci=0.95):
    """Two-stage batch-level bootstrap CI for a real-data coverage/width
    point estimate (e.g. a row of Table 2), which otherwise comes from a
    single fixed train/cal/test split with no sense of how much it would
    move under a different draw.

    Resamples calibration batches AND test batches independently, with
    replacement, B times -- BATCHES, not patients, are the resampling unit
    throughout, matching the exchangeable unit split-conformal validity is
    defined over (resampling at the patient level would silently discard
    the within-batch correlation this paper's method is built to respect).

    The frozen classifier that produced cal_p_hat/test_p_hat is NOT refit
    per replicate: Theorem 1's guarantee is about calibrating a FIXED
    p_hat, so this should capture calibration- and test-sampling noise
    only, not model-refitting noise -- a different, larger question this
    estimate isn't making a claim about.

    method_fn(cal_batches, cal_p_hat, test_batches, test_p_hat) ->
    (coverage, width) -- one call per replicate. Build this once per
    method via methods.make_method_scorers(alpha) and reuse the SAME
    closure for the point estimate and every bootstrap replicate, so they
    can't silently diverge.

    Returns percentile-method CIs (robust to the skew coverage rates near
    0 or 1 can have, unlike a normal-approximation CI) plus the bootstrap
    mean, for both coverage and width.
    """
    rng = np.random.default_rng(seed)
    n_cal, n_test = len(cal_batches), len(test_batches)
    cov_samples = np.empty(B)
    width_samples = np.empty(B)
    for i in range(B):
        cal_idx = rng.integers(0, n_cal, size=n_cal)
        test_idx = rng.integers(0, n_test, size=n_test)
        cal_b = [cal_batches[j] for j in cal_idx]
        cal_p = [cal_p_hat[j] for j in cal_idx]
        test_b = [test_batches[j] for j in test_idx]
        test_p = [test_p_hat[j] for j in test_idx]
        cov_samples[i], width_samples[i] = method_fn(cal_b, cal_p, test_b, test_p)

    cov_lo, cov_hi = _percentile_ci(cov_samples, ci)
    w_lo, w_hi = _percentile_ci(width_samples, ci)
    return {
        "coverage_boot_mean": float(cov_samples.mean()), "coverage_ci": (cov_lo, cov_hi),
        "width_boot_mean": float(width_samples.mean()), "width_ci": (w_lo, w_hi),
        "B": B,
    }


def block_bootstrap_aci_ci(day_batches_list, day_p_hat_list, day_is_test, run_fn,
                            B=500, block_len=None, seed=0, ci=0.95):
    """Moving-block bootstrap CI for an online-adaptive (ACI) procedure's
    realized test-period coverage/width -- appropriate for day-sequential
    data, where a plain i.i.d. batch resample (bootstrap_coverage_width)
    would break the local day-to-day dependence the ACI recursion relies
    on (alpha_t and the running nonconformity pool both carry over from
    recent days).

    Resamples CONTIGUOUS blocks of days with replacement (preserving
    in-block order exactly, so short-range day-to-day dependence survives
    within a block) to build a bootstrap day-sequence of the same length,
    reruns run_fn on it from the same fixed warm start, and scores only
    the days flagged is_test in that SAME bootstrap draw -- standard
    moving-block bootstrap (Kunsch 1989) adapted to a day-indexed adaptive
    procedure instead of a plain time series.

    day_batches_list / day_p_hat_list: per-day batch/p_hat lists,
        chronological, covering the full walk-forward sequence (the fixed
        seed pool that warm-starts the recursion is closed over inside
        run_fn, not resampled here -- it's history that already happened,
        not part of what we're asking "how much would this vary").
    day_is_test: boolean array, same length, True for days in the scored
        test period.
    run_fn(boot_day_batches, boot_day_p_hat) -> flat bounds array (one
        entry per batch, in the same order as boot_day_batches flattened)
        -- a closure the caller builds around adaptive.run_aci_batch_
        conformal with the fixed seed pool, alpha and gamma already bound.
    block_len: contiguous block length in days; defaults to the standard
        block-bootstrap heuristic round(n_days ** (1/3)), floored at 2.
        With only a handful of days available (as in the shorter of the
        two real datasets here), the resulting CI is necessarily wide --
        that's a real reflection of how little independent day-level
        information exists, not an artifact of the method.
    """
    n_days = len(day_batches_list)
    day_is_test = np.asarray(day_is_test, dtype=bool)
    if block_len is None:
        block_len = max(2, round(n_days ** (1 / 3)))
    block_len = min(block_len, n_days)
    max_start = max(0, n_days - block_len)
    n_blocks_needed = int(np.ceil(n_days / block_len))

    rng = np.random.default_rng(seed)
    cov_samples, width_samples = [], []
    for _ in range(B):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        boot_idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n_days]
        boot_days = [day_batches_list[j] for j in boot_idx]
        boot_p_hat = [day_p_hat_list[j] for j in boot_idx]
        boot_is_test = day_is_test[boot_idx]

        bounds = np.asarray(run_fn(boot_days, boot_p_hat))
        S = np.concatenate([[np.sum(b["Y"]) for b in day] for day in boot_days])
        test_mask = np.concatenate([[flag] * len(day) for flag, day in zip(boot_is_test, boot_days)])
        S_test, bounds_test = S[test_mask], bounds[test_mask]
        if len(S_test) == 0:
            continue
        cov_samples.append(np.mean(S_test <= bounds_test))
        width_samples.append(np.mean(bounds_test))

    cov_samples, width_samples = np.array(cov_samples), np.array(width_samples)
    cov_lo, cov_hi = _percentile_ci(cov_samples, ci)
    w_lo, w_hi = _percentile_ci(width_samples, ci)
    return {
        "coverage_boot_mean": float(cov_samples.mean()), "coverage_ci": (cov_lo, cov_hi),
        "width_boot_mean": float(width_samples.mean()), "width_ci": (w_lo, w_hi),
        "B": len(cov_samples), "block_len": block_len,
    }
