"""Candidate upper-bound constructions for S_b = total batch attendance.

All three take a list of batches (each with p_hat_i already computed by a
FROZEN classifier) and produce, for each batch, an upper bound U_b such that
the claim is P(S_b <= U_b) >= 1 - alpha.
"""
import numpy as np


def naive_parametric_bound(batches, p_hat_list, alpha=0.1):
    """Status-quo OR baseline: Poisson-binomial Normal approximation,
    assuming independent Bernoullis (ignores within-batch correlation).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha)
    bounds = []
    for batch, p_hat in zip(batches, p_hat_list):
        mu = np.sum(p_hat)
        sigma2 = np.sum(p_hat * (1 - p_hat))
        bounds.append(mu + z * np.sqrt(sigma2))
    return np.array(bounds)


def dro_cantelli_bound(batches, p_hat_list, alpha=0.1):
    """Classical distributionally-robust bound (Cantelli / one-sided
    Chebyshev): P(Z >= mu + k*sigma) <= 1/(1+k^2), so k=sqrt((1-alpha)/alpha)
    gives a worst-case-over-DISTRIBUTIONAL-SHAPE guarantee using only mean
    and variance -- no Gaussian/CLT assumption.

    IMPORTANT, verified by a failing test before this docstring was written:
    this bound uses the PLUG-IN Poisson-binomial variance (same formula as
    naive_parametric_bound), which is the variance UNDER INDEPENDENCE. That
    means Cantelli's shape-free guarantee is valid, but only for the WRONG
    (too-small) variance under real within-batch correlation -- it inherits
    the SAME failure mode as the naive Normal approximation, just with a
    bigger safety margin (k=3 at alpha=0.1 vs the Normal's z=1.28) that
    correlation can still overwhelm at high enough tau. This is a genuine,
    citable point: "distribution-free" in the classical moment-based OR
    sense does NOT by itself protect against variance misspecification --
    see dro_cantelli_calibrated_bound for the fair, corrected comparator.
    """
    k = np.sqrt((1 - alpha) / alpha)
    bounds = []
    for batch, p_hat in zip(batches, p_hat_list):
        mu, sigma = _mu_sigma(p_hat)
        bounds.append(mu + k * sigma)
    return np.array(bounds)


def fit_dro_cantelli_calibrated(cal_batches, cal_p_hat_list, alpha=0.1):
    """The FAIR DRO comparator: still makes NO distributional-shape
    assumption (Cantelli's worst-case guarantee), but estimates sigma from
    HISTORICAL batch-level residuals (like our conformal method does)
    rather than the naive independent-Bernoulli plug-in formula -- so it
    correctly picks up real within-batch correlation instead of silently
    ignoring it. This is what a careful DRO practitioner should actually
    do: estimate the ambiguity set's moments from data, don't assume them.
    """
    residuals = []
    for batch, p_hat in zip(cal_batches, cal_p_hat_list):
        mu = np.sum(p_hat)
        residuals.append(np.sum(batch["Y"]) - mu)
    sigma_hist = np.std(np.array(residuals), ddof=1)
    return {"sigma_hist": sigma_hist, "alpha": alpha}


def dro_cantelli_calibrated_bound(fit, test_batches, test_p_hat_list):
    k = np.sqrt((1 - fit["alpha"]) / fit["alpha"])
    bounds = []
    for batch, p_hat in zip(test_batches, test_p_hat_list):
        mu = np.sum(p_hat)
        bounds.append(mu + k * fit["sigma_hist"])
    return np.array(bounds)


def _mu_sigma(p_hat):
    mu = np.sum(p_hat)
    sigma2 = np.sum(p_hat * (1 - p_hat))
    return mu, np.sqrt(max(sigma2, 1e-8))


def fit_batch_conformal(cal_batches, cal_p_hat_list, alpha=0.1):
    """Our method: studentized nonconformity score on HISTORICAL, COMPLETE
    calibration batches, batches (not patients) as the exchangeable unit.
    """
    R = []
    for batch, p_hat in zip(cal_batches, cal_p_hat_list):
        mu, sigma = _mu_sigma(p_hat)
        S_b = np.sum(batch["Y"])
        R.append((S_b - mu) / sigma)
    R = np.array(R)
    n = len(R)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)  # guard k=0 (only reachable at alpha>=1, never used here, but adaptive.py already guards this -- match it)
    q_hat = np.sort(R)[k - 1]
    return {"q_hat": q_hat, "R_cal": R}


def batch_conformal_bound(fit, test_batches, test_p_hat_list):
    bounds = []
    for batch, p_hat in zip(test_batches, test_p_hat_list):
        mu, sigma = _mu_sigma(p_hat)
        bounds.append(mu + fit["q_hat"] * sigma)
    return np.array(bounds)


def fit_bonferroni_individual(cal_patients_p, cal_patients_y, n_bins=10):
    """Provably-valid-under-arbitrary-dependence baseline: bins patients by
    predicted probability (a simple, standard conformal-classification
    device: nonconformity score for a patient is |Y - p_hat|; calibration
    is done WITHIN probability bins so the bound adapts to how
    well-calibrated the classifier is in that range) and stores each bin's
    SORTED nonconformity scores -- not a single precomputed quantile.

    The Bonferroni correction level alpha/N_b depends on N_b, the size of
    the SPECIFIC test batch being bounded, which isn't known yet at fit
    time (real batch sizes here range from single digits to several
    hundred). Storing raw per-bin scores lets bonferroni_bound() compute
    the right quantile for each test batch's own size at bound-construction
    time, tighter than calibrating once against a single global worst-case
    N while remaining exactly as valid for every batch -- see
    bonferroni_bound_single.
    """
    bins = np.quantile(cal_patients_p, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    bin_idx = np.digitize(cal_patients_p, bins[1:-1])
    nonconformity = np.abs(cal_patients_y - cal_patients_p)
    bin_scores = {b: np.sort(nonconformity[bin_idx == b]) for b in range(n_bins)}
    return {"bins": bins, "bin_scores": bin_scores}


def _bonferroni_quantile(sorted_scores, alpha_level):
    """(1 - alpha_level) upper quantile of a bin's calibration nonconformity
    scores, via the standard finite-sample conformal correction (the
    ceil((m+1)(1-alpha_level))-th order statistic). Falls back to the
    VACUOUS bound (1.0 -- the maximum value |Y-p_hat| can take) when the
    bin has too few calibration points to support the requested level
    (k > m): using the largest OBSERVED score instead, as an earlier
    version of this code did, is not a valid (1-alpha_level) quantile
    estimate from only m samples and silently under-covers. This is the
    same convention already used for empty bins (m=0 -> 1.0), applied
    consistently instead of only at that one edge.
    """
    m = len(sorted_scores)
    if m == 0:
        return 1.0
    k = int(np.ceil((m + 1) * (1 - alpha_level)))
    if k > m:
        return 1.0
    return sorted_scores[k - 1]


def bonferroni_bound_single(fit, p_hat, alpha=0.1):
    """Per-patient upper bound on Y_i (capped at 1) for ONE batch/candidate
    pool of size N: U_i = min(1, p_hat_i + q_i), where q_i is the bin's
    (1 - alpha/N) nonconformity quantile. The correction is keyed to N,
    THIS pool's own size, so P(exists i in this N-patient set: Y_i > U_i)
    <= N * (alpha/N) = alpha regardless of how large any other batch in
    the data happens to be. Returns sum(U_i), the batch-level bound.

    Used directly (not just via bonferroni_bound) by decision.py's
    max_bookable, which evaluates bound_fn on growing prefixes of a
    candidate pool -- each prefix is its own "N" for this purpose.
    """
    n = len(p_hat)
    if n == 0:
        return 0.0
    alpha_level = alpha / n
    bin_idx = np.digitize(p_hat, fit["bins"][1:-1])
    q = np.array([_bonferroni_quantile(fit["bin_scores"][b], alpha_level) for b in bin_idx])
    return float(np.sum(np.clip(p_hat + q, 0, 1)))


def bonferroni_bound(fit, test_batches, test_p_hat_list, alpha=0.1):
    """Per-test-batch Bonferroni bound: each batch's own size sets its own
    correction level (alpha / len(batch)) via bonferroni_bound_single,
    tighter than calibrating once against a single global worst-case batch
    size while remaining exactly as valid for every batch.
    """
    return np.array([bonferroni_bound_single(fit, p_hat, alpha=alpha) for p_hat in test_p_hat_list])


def make_method_scorers(alpha, n_bins=8):
    """Returns {method_name: fn(cal_batches, cal_p_hat, test_batches,
    test_p_hat) -> (coverage, width)} for all five methods -- ONE shared
    implementation of "how does each method turn a (calibration, test)
    split into a (coverage, width) point estimate", reused both for the
    single point estimate reported on the real data and for every
    replicate inside a bootstrap CI (stats_utils.bootstrap_coverage_width),
    so the point estimate and its own CI can never silently drift apart
    from two independently-maintained code paths.
    """
    def _score(bounds, batches):
        S = np.array([np.sum(b["Y"]) for b in batches])
        return float(np.mean(S <= bounds)), float(np.mean(bounds))

    def naive_fn(cal_b, cal_p, test_b, test_p):
        return _score(naive_parametric_bound(test_b, test_p, alpha=alpha), test_b)

    def dro_plugin_fn(cal_b, cal_p, test_b, test_p):
        return _score(dro_cantelli_bound(test_b, test_p, alpha=alpha), test_b)

    def dro_calib_fn(cal_b, cal_p, test_b, test_p):
        fit = fit_dro_cantelli_calibrated(cal_b, cal_p, alpha=alpha)
        return _score(dro_cantelli_calibrated_bound(fit, test_b, test_p), test_b)

    def ours_fn(cal_b, cal_p, test_b, test_p):
        fit = fit_batch_conformal(cal_b, cal_p, alpha=alpha)
        return _score(batch_conformal_bound(fit, test_b, test_p), test_b)

    def bonferroni_fn(cal_b, cal_p, test_b, test_p):
        cal_patients_p = np.concatenate(cal_p)
        cal_patients_y = np.concatenate([b["Y"] for b in cal_b])
        fit = fit_bonferroni_individual(cal_patients_p, cal_patients_y, n_bins=n_bins)
        return _score(bonferroni_bound(fit, test_b, test_p, alpha=alpha), test_b)

    return {"naive": naive_fn, "dro_plugin": dro_plugin_fn, "dro_calib": dro_calib_fn,
            "ours": ours_fn, "bonferroni": bonferroni_fn}
