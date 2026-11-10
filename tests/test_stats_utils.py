import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batch_risk_control.stats_utils import (
    paired_comparison, coverage_vs_nominal, format_p,
    bootstrap_coverage_width, block_bootstrap_aci_ci,
)


def test_paired_comparison_detects_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, size=50)   # consistently smaller
    b = rng.normal(15, 1, size=50)
    result = paired_comparison(a, b, "A", "B", alternative="less")
    assert result["mean_diff"] < 0
    assert result["wilcoxon_p"] < 0.001
    assert result["ttest_p"] < 0.001
    assert result["ci95"][1] < 0  # entire 95% CI below zero -> confident A < B


def test_paired_comparison_no_difference_gives_high_p():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 1, size=200)
    b = a + rng.normal(0, 0.01, size=200)  # essentially identical
    result = paired_comparison(a, b, "A", "B", alternative="two-sided")
    assert result["wilcoxon_p"] > 0.05 or abs(result["mean_diff"]) < 0.05


def test_paired_comparison_identical_arrays_no_crash():
    a = np.full(10, 5.0)
    result = paired_comparison(a, a, "A", "B", alternative="two-sided")
    assert result["mean_diff"] == 0.0
    assert result["wilcoxon_p"] == 1.0


def test_coverage_vs_nominal_detects_significant_shortfall():
    rng = np.random.default_rng(2)
    coverage = rng.normal(0.70, 0.02, size=40)  # far below 0.90
    result = coverage_vs_nominal(coverage, nominal=0.90)
    assert result["mean"] < 0.90
    assert result["ttest_p"] < 0.001
    assert result["ci95"][1] < 0.90


def test_coverage_vs_nominal_near_target_not_significant():
    rng = np.random.default_rng(3)
    coverage = rng.normal(0.901, 0.01, size=40)  # essentially at target
    result = coverage_vs_nominal(coverage, nominal=0.90)
    assert result["ttest_p"] > 0.05


def test_format_p():
    assert format_p(0.0001) == "<0.001"
    assert format_p(0.05) == "0.050"
    assert format_p(0.9999) == "1.000"


def _fake_batches(rng, n, size=10, p=0.5):
    batches, p_hats = [], []
    for _ in range(n):
        batches.append({"Y": rng.binomial(1, p, size=size)})
        p_hats.append(np.full(size, p))
    return batches, p_hats


def _fixed_margin_scorer(margin):
    def method_fn(cal_b, cal_p, test_b, test_p):
        bounds = np.array([np.sum(p) + margin for p in test_p])
        S = np.array([np.sum(b["Y"]) for b in test_b])
        return np.mean(S <= bounds), np.mean(bounds)
    return method_fn


def test_bootstrap_coverage_width_mean_close_to_point_estimate():
    """Bootstrap mean coverage/width should land close to the plain point
    estimate computed on the un-resampled data -- a sanity check that
    resampling batches with replacement doesn't introduce systematic bias."""
    rng = np.random.default_rng(0)
    cal_b, cal_p = _fake_batches(rng, n=200, size=10, p=0.4)
    test_b, test_p = _fake_batches(rng, n=200, size=10, p=0.4)
    method_fn = _fixed_margin_scorer(margin=2.0)

    point_cov, point_width = method_fn(cal_b, cal_p, test_b, test_p)
    result = bootstrap_coverage_width(cal_b, cal_p, test_b, test_p, method_fn, B=300, seed=1)

    assert abs(result["coverage_boot_mean"] - point_cov) < 0.05
    assert abs(result["width_boot_mean"] - point_width) < 0.5
    assert result["coverage_ci"][0] <= result["coverage_boot_mean"] <= result["coverage_ci"][1]
    assert result["width_ci"][0] <= result["width_boot_mean"] <= result["width_ci"][1]


def test_bootstrap_coverage_width_ci_narrows_with_more_batches():
    """More test batches -> less sampling noise -> a narrower CI, holding
    the underlying data-generating process fixed."""
    method_fn = _fixed_margin_scorer(margin=2.0)

    rng_small = np.random.default_rng(2)
    cal_b, cal_p = _fake_batches(rng_small, n=20, size=10, p=0.4)
    test_b, test_p = _fake_batches(rng_small, n=20, size=10, p=0.4)
    small = bootstrap_coverage_width(cal_b, cal_p, test_b, test_p, method_fn, B=500, seed=3)

    rng_large = np.random.default_rng(2)
    cal_b_l, cal_p_l = _fake_batches(rng_large, n=500, size=10, p=0.4)
    test_b_l, test_p_l = _fake_batches(rng_large, n=500, size=10, p=0.4)
    large = bootstrap_coverage_width(cal_b_l, cal_p_l, test_b_l, test_p_l, method_fn, B=500, seed=3)

    ci_width_small = small["coverage_ci"][1] - small["coverage_ci"][0]
    ci_width_large = large["coverage_ci"][1] - large["coverage_ci"][0]
    assert ci_width_large < ci_width_small


def test_bootstrap_coverage_width_resamples_batches_not_patients():
    """A method_fn that is sensitive to WHICH batches were drawn (not just
    how many patients) should show bootstrap variability -- confirms
    resampling happens at the batch level, not by pooling and reshuffling
    patients (which would silently discard within-batch structure)."""
    rng = np.random.default_rng(9)
    # one giant batch and many tiny batches -- patient-level resampling
    # would blend them; batch-level resampling must sometimes include the
    # giant batch and sometimes not, producing real width variability.
    test_b = [{"Y": np.ones(200, dtype=int)}] + [{"Y": np.zeros(2, dtype=int)} for _ in range(50)]
    test_p = [np.full(200, 0.5)] + [np.full(2, 0.5) for _ in range(50)]
    cal_b, cal_p = test_b, test_p
    method_fn = _fixed_margin_scorer(margin=0.0)

    result = bootstrap_coverage_width(cal_b, cal_p, test_b, test_p, method_fn, B=500, seed=10)
    assert result["width_ci"][1] > result["width_ci"][0] + 1.0, (
        "width should vary meaningfully across replicates when one batch dominates the total"
    )


def test_block_bootstrap_aci_ci_basic_sanity():
    """A generous fixed-margin run_fn should show near-1.0 bootstrap
    coverage, with the CI bracketing the bootstrap mean and a sane
    (>=2 day) block length chosen by the default heuristic."""
    rng = np.random.default_rng(5)
    n_days = 30
    day_batches = [[{"Y": rng.binomial(1, 0.4, size=8)}] for _ in range(n_days)]
    day_p_hat = [[np.full(8, 0.4)] for _ in range(n_days)]
    is_test = np.array([False] * 15 + [True] * 15)

    def run_fn(boot_days, boot_p_hat):
        return np.array([np.sum(p) + 3.0 for day, p_day in zip(boot_days, boot_p_hat) for p in p_day])

    result = block_bootstrap_aci_ci(day_batches, day_p_hat, is_test, run_fn, B=200, seed=6)
    assert result["coverage_boot_mean"] > 0.95
    assert result["coverage_ci"][0] <= result["coverage_boot_mean"] <= result["coverage_ci"][1]
    assert result["block_len"] >= 2


def test_block_bootstrap_aci_ci_respects_custom_block_len():
    rng = np.random.default_rng(7)
    n_days = 20
    day_batches = [[{"Y": rng.binomial(1, 0.5, size=5)}] for _ in range(n_days)]
    day_p_hat = [[np.full(5, 0.5)] for _ in range(n_days)]
    is_test = np.array([True] * n_days)

    def run_fn(boot_days, boot_p_hat):
        return np.array([np.sum(p) + 1.0 for day, p_day in zip(boot_days, boot_p_hat) for p in p_day])

    result = block_bootstrap_aci_ci(day_batches, day_p_hat, is_test, run_fn, B=50, block_len=4, seed=8)
    assert result["block_len"] == 4


def test_block_bootstrap_aci_ci_preserves_within_block_order():
    """Each bootstrap day-sequence must be built from CONTIGUOUS runs of
    the original day order (moving-block bootstrap), not an i.i.d. day
    shuffle -- every run of block_len positions in a bootstrap draw must
    itself be block_len consecutive original day-indices."""
    n_days = 25
    day_batches = [[{"Y": np.array([1])}] for _ in range(n_days)]
    day_p_hat = [[np.array([0.5])] for _ in range(n_days)]
    is_test = np.array([True] * n_days)
    # every day's CONTENT is identical by design (only order/contiguity is
    # under test), so recover which original day slot was drawn via object
    # identity rather than == (content equality can't distinguish them)
    id_to_idx = {id(day_batches[i]): i for i in range(n_days)}
    seen_draws = []

    def run_fn(boot_days, boot_p_hat):
        seen_draws.append([id_to_idx[id(d)] for d in boot_days])
        return np.array([np.sum(p) for day in boot_p_hat for p in day])

    block_bootstrap_aci_ci(day_batches, day_p_hat, is_test, run_fn, B=20, block_len=5, seed=15)
    for idx in seen_draws:
        for k in range(0, len(idx), 5):
            block = idx[k:k + 5]
            if len(block) == 5:
                assert block == list(range(block[0], block[0] + 5)), f"block {block} is not contiguous"
