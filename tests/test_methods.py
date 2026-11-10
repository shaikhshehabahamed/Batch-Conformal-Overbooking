import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batch_risk_control.dgp import generate_batches, predicted_p
from batch_risk_control.methods import (
    naive_parametric_bound, fit_batch_conformal, batch_conformal_bound,
    fit_bonferroni_individual, bonferroni_bound, bonferroni_bound_single,
    _bonferroni_quantile, dro_cantelli_bound,
    fit_dro_cantelli_calibrated, dro_cantelli_calibrated_bound,
)


def test_dro_cantelli_matches_hand_computation_and_exceeds_naive():
    """alpha=0.1 -> k=sqrt(0.9/0.1)=3.0 exactly; must be wider than the
    Normal-approximation naive bound since it makes no shape assumption."""
    batches = [{"Y": np.array([1, 0])}]
    p_hat_list = [np.array([0.5, 0.5])]
    dro = dro_cantelli_bound(batches, p_hat_list, alpha=0.1)
    expected = 1.0 + 3.0 * np.sqrt(0.5)
    np.testing.assert_allclose(dro, [expected], atol=1e-8)
    naive = naive_parametric_bound(batches, p_hat_list, alpha=0.1)
    assert dro[0] > naive[0]


def test_dro_cantelli_plugin_also_fails_under_strong_enough_correlation():
    """Documents a real, verified finding: the PLUG-IN Cantelli bound uses
    the same misspecified (independence-assuming) variance as the naive
    baseline, so its shape-free guarantee does NOT protect against variance
    underestimation -- it degrades under strong correlation too, just with
    a bigger safety margin than the Normal approximation (fails later, not
    never). This motivates dro_cantelli_calibrated_bound as the fair
    comparator instead of trusting the plug-in version at face value."""
    rng = np.random.default_rng(6)
    batches = generate_batches(n_batches=400, batch_size_range=(20, 80), tau=2.0, rng=rng)
    p_hat_list = [predicted_p(b, batches[0]["beta"]) for b in batches]
    test = batches[300:]
    test_p = p_hat_list[300:]
    S_test = np.array([np.sum(b["Y"]) for b in test])
    dro_bounds = dro_cantelli_bound(test, test_p, alpha=0.1)
    cov = np.mean(S_test <= dro_bounds)
    assert cov < 0.90, "expected the plug-in Cantelli bound to also undercover under strong tau=2.0 correlation"


def test_dro_cantelli_calibrated_is_valid_under_correlation():
    """The historical-variance-corrected Cantelli bound should hold at or
    above nominal coverage even under strong induced correlation, since it
    no longer relies on the independence-assuming plug-in variance."""
    rng = np.random.default_rng(7)
    batches = generate_batches(n_batches=500, batch_size_range=(20, 80), tau=2.0, rng=rng)
    p_hat_list = [predicted_p(b, batches[0]["beta"]) for b in batches]
    cal, test = batches[:350], batches[350:]
    cal_p, test_p = p_hat_list[:350], p_hat_list[350:]
    S_test = np.array([np.sum(b["Y"]) for b in test])
    fit = fit_dro_cantelli_calibrated(cal, cal_p, alpha=0.1)
    bounds = dro_cantelli_calibrated_bound(fit, test, test_p)
    cov = np.mean(S_test <= bounds)
    assert cov >= 0.85, f"calibrated Cantelli should be valid (~>=90%) under correlation, got {cov:.3f}"


def test_dgp_shapes_and_ranges():
    rng = np.random.default_rng(0)
    batches = generate_batches(n_batches=20, batch_size_range=(10, 15), n_features=3, tau=1.0, rng=rng)
    assert len(batches) == 20
    for b in batches:
        n_i = b["X"].shape[0]
        assert 10 <= n_i <= 15
        assert b["Y"].shape == (n_i,)
        assert set(np.unique(b["Y"])).issubset({0, 1})
        assert np.all((b["true_p"] > 0) & (b["true_p"] < 1))


def test_tau_zero_induces_no_batch_effect_variance():
    """With tau=0, eta must be exactly 0 for every batch (no shared effect)."""
    rng = np.random.default_rng(1)
    batches = generate_batches(n_batches=50, tau=0.0, rng=rng)
    etas = [b["eta"] for b in batches]
    assert np.allclose(etas, 0.0)


def test_naive_bound_matches_hand_computation():
    """2 patients, p_hat=[0.5, 0.5]: mu=1.0, sigma^2=0.25+0.25=0.5."""
    from scipy.stats import norm
    batches = [{"Y": np.array([1, 0])}]
    p_hat_list = [np.array([0.5, 0.5])]
    bound = naive_parametric_bound(batches, p_hat_list, alpha=0.1)
    expected = 1.0 + norm.ppf(0.9) * np.sqrt(0.5)
    np.testing.assert_allclose(bound, [expected], atol=1e-8)


def test_batch_conformal_calibration_is_monotone_in_alpha():
    """Smaller alpha (higher confidence) must give a wider (or equal) bound."""
    rng = np.random.default_rng(2)
    batches = generate_batches(n_batches=200, tau=1.0, rng=rng)
    p_hat_list = [predicted_p(b, batches[0]["beta"]) for b in batches]
    cal, test = batches[:150], batches[150:]
    cal_p, test_p = p_hat_list[:150], p_hat_list[150:]

    fit_90 = fit_batch_conformal(cal, cal_p, alpha=0.10)
    fit_99 = fit_batch_conformal(cal, cal_p, alpha=0.01)
    bound_90 = batch_conformal_bound(fit_90, test, test_p)
    bound_99 = batch_conformal_bound(fit_99, test, test_p)
    assert np.all(bound_99 >= bound_90 - 1e-9)


def test_bonferroni_bound_never_exceeds_batch_size():
    """Sum of per-patient bounds, each clipped to [0,1], can't exceed N."""
    rng = np.random.default_rng(3)
    batches = generate_batches(n_batches=200, tau=1.0, rng=rng)
    p_hat_list = [predicted_p(b, batches[0]["beta"]) for b in batches]
    cal, test = batches[:150], batches[150:]
    cal_p, test_p = p_hat_list[:150], p_hat_list[150:]
    cal_patients_p = np.concatenate(cal_p)
    cal_patients_y = np.concatenate([b["Y"] for b in cal])

    fit = fit_bonferroni_individual(cal_patients_p, cal_patients_y, n_bins=8)
    bounds = bonferroni_bound(fit, test, test_p, alpha=0.10)
    sizes = np.array([len(b["Y"]) for b in test])
    assert np.all(bounds <= sizes + 1e-9)


def test_bonferroni_quantile_matches_hand_computation_when_data_sufficient():
    """m=20 calibration points, alpha_level=0.1 -> k=ceil(21*0.9)=19, well
    within range -- must return the 19th order statistic exactly, same rule
    the original (pre-fix) code used in this regime."""
    scores = np.sort(np.arange(1, 21) / 20.0)  # 0.05, 0.10, ..., 1.00
    q = _bonferroni_quantile(scores, alpha_level=0.1)
    assert q == scores[18]  # k=19 -> index 18


def test_bonferroni_quantile_vacuous_when_calibration_data_insufficient():
    """m=5 points can't support alpha_level=0.01 (k=ceil(6*0.99)=6 > 5=m):
    the ONLY valid answer from 5 samples at that level is the vacuous
    (1.0) bound. Returning the max OBSERVED score instead (the bug this
    replaces) would silently under-cover, since 5 samples cannot support a
    99% quantile claim."""
    scores = np.sort(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    q = _bonferroni_quantile(scores, alpha_level=0.01)
    assert q == 1.0


def test_bonferroni_quantile_empty_bin_is_vacuous():
    assert _bonferroni_quantile(np.array([]), alpha_level=0.1) == 1.0


def test_bonferroni_per_batch_correction_tighter_for_smaller_batches():
    """The whole point of the fix: a smaller test batch should get a
    TIGHTER (less conservative) per-patient correction than a larger one,
    because alpha/N_b depends on N_b, not on the largest batch ever seen.
    Tested directly on the quantile function with scores that have real
    spread -- a p_hat=0.5 nonconformity distribution would collapse to a
    single constant (|Y-0.5|=0.5 regardless of Y), which trivially gives
    the same "quantile" at any level and would defeat this test."""
    rng = np.random.default_rng(11)
    scores = np.sort(rng.uniform(0, 1, size=2000))
    alpha = 0.10
    q_small_batch = _bonferroni_quantile(scores, alpha_level=alpha / 5)    # N=5
    q_large_batch = _bonferroni_quantile(scores, alpha_level=alpha / 300)  # N=300
    assert q_small_batch < q_large_batch, (
        "a smaller batch's own-size correction should be less conservative "
        "than a larger batch's, from the same calibration scores"
    )


def test_bonferroni_global_max_correction_is_upper_bound_on_per_batch_fix():
    """Sanity check against the previous behavior: for any single test
    batch, the per-batch-corrected bound (alpha/N_b) must be <= what the
    old global-max scheme (alpha/N_max, N_max >= N_b) would have given,
    since a smaller alpha_level (stricter correction) can only produce an
    equal-or-larger quantile. This is the formal reason the fix is a
    genuine tightening, not just a different number."""
    rng = np.random.default_rng(12)
    cal_patients_p = rng.uniform(0, 1, size=5_000)
    cal_patients_y = rng.binomial(1, cal_patients_p)
    fit = fit_bonferroni_individual(cal_patients_p, cal_patients_y, n_bins=8)

    p_hat = rng.uniform(0.2, 0.8, size=30)  # N_b=30, well below an assumed N_max
    n_max = 300
    bound_per_batch = bonferroni_bound_single(fit, p_hat, alpha=0.10)
    bound_global_max = bonferroni_bound_single(fit, p_hat, alpha=0.10 * len(p_hat) / n_max)  # emulates alpha/n_max
    assert bound_per_batch <= bound_global_max + 1e-9


def test_bonferroni_per_batch_correction_remains_valid_under_correlation():
    """End-to-end validity check: the per-batch-corrected Bonferroni bound
    must still hold at or above nominal coverage under real within-batch
    correlation and heterogeneous batch sizes -- the fix must tighten the
    bound without ever breaking the union-bound guarantee it exists for."""
    rng = np.random.default_rng(13)
    batches = generate_batches(n_batches=600, batch_size_range=(5, 300), tau=1.5, rng=rng)
    p_hat_list = [predicted_p(b, batches[0]["beta"]) for b in batches]
    cal, test = batches[:400], batches[400:]
    cal_p, test_p = p_hat_list[:400], p_hat_list[400:]
    cal_patients_p = np.concatenate(cal_p)
    cal_patients_y = np.concatenate([b["Y"] for b in cal])
    S_test = np.array([np.sum(b["Y"]) for b in test])

    fit = fit_bonferroni_individual(cal_patients_p, cal_patients_y, n_bins=8)
    bounds = bonferroni_bound(fit, test, test_p, alpha=0.10)
    cov = np.mean(S_test <= bounds)
    assert cov >= 0.90, f"Bonferroni must remain provably valid after the per-batch fix, got {cov:.3f}"


def test_our_method_matches_naive_width_when_no_correlation():
    """With tau=0 the naive independence assumption is exactly correct, so
    our conformal correction should add negligible extra width (adapts
    down rather than always padding conservatively)."""
    rng = np.random.default_rng(4)
    batches = generate_batches(n_batches=400, batch_size_range=(20, 80), tau=0.0, rng=rng)
    p_hat_list = [predicted_p(b, batches[0]["beta"]) for b in batches]
    cal, test = batches[:300], batches[300:]
    cal_p, test_p = p_hat_list[:300], p_hat_list[300:]

    naive = naive_parametric_bound(test, test_p, alpha=0.10)
    fit = fit_batch_conformal(cal, cal_p, alpha=0.10)
    ours = batch_conformal_bound(fit, test, test_p)

    rel_diff = abs(np.mean(ours) - np.mean(naive)) / np.mean(naive)
    assert rel_diff < 0.10, f"expected near-parity at tau=0, got {rel_diff:.3f} relative gap"
