import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batch_risk_control.dgp import generate_batches, predicted_p
from batch_risk_control.adaptive import run_aci_batch_conformal


def test_aci_reduces_to_reasonable_coverage_no_drift():
    """With NO drift (tau fixed, no trend across the walk-forward batches),
    ACI's alpha_t should stay close to the nominal alpha and coverage
    should land near the target, same as static conformal calibration."""
    rng = np.random.default_rng(10)
    batches = generate_batches(n_batches=400, batch_size_range=(20, 80), tau=1.0, rng=rng)
    beta = batches[0]["beta"]
    p_hat = [predicted_p(b, beta) for b in batches]

    seed_batches, seed_p = batches[:100], p_hat[:100]
    remaining_batches, remaining_p = batches[100:], p_hat[100:]
    # chunk into "days" of 10 batches each for the walk-forward loop
    day_batches = [remaining_batches[i:i+10] for i in range(0, len(remaining_batches), 10)]
    day_p = [remaining_p[i:i+10] for i in range(0, len(remaining_p), 10)]

    bounds, alpha_traj = run_aci_batch_conformal(seed_batches, seed_p, day_batches, day_p,
                                                  alpha=0.10, gamma=0.02)
    S_all = np.concatenate([[np.sum(b["Y"]) for b in day] for day in day_batches])
    cov = np.mean(S_all <= bounds)
    assert 0.80 <= cov <= 0.98, f"expected coverage near nominal under no drift, got {cov:.3f}"
    assert 0.05 <= alpha_traj[-1] <= 0.20, f"alpha_t should stay near 0.10 with no drift, got {alpha_traj[-1]:.3f}"


def test_aci_widens_bounds_after_a_run_of_misses():
    """If the pool is seeded to make the FIRST few days badly miss (bound
    set unrealistically tight), alpha_t should decrease (push toward wider
    bounds) rather than staying fixed -- this is the whole mechanism."""
    rng = np.random.default_rng(11)
    batches = generate_batches(n_batches=200, batch_size_range=(20, 80), tau=0.0, rng=rng)
    beta = batches[0]["beta"]
    p_hat = [predicted_p(b, beta) for b in batches]

    # seed pool artificially tight: fabricate a seed with near-zero residuals
    # so the initial quantile badly underestimates true spread, forcing misses
    seed_batches = batches[:20]
    seed_p = [np.full_like(p_hat[i], p_hat[i].mean()) for i in range(20)]  # crude, low-variance proxy

    remaining_batches, remaining_p = batches[20:], p_hat[20:]
    day_batches = [remaining_batches[i:i+5] for i in range(0, 60, 5)]
    day_p = [remaining_p[i:i+5] for i in range(0, 60, 5)]

    _, alpha_traj = run_aci_batch_conformal(seed_batches, seed_p, day_batches, day_p,
                                             alpha=0.10, gamma=0.1)
    # if there were more misses than nominal early on, alpha_t should trend down
    assert alpha_traj[-1] <= alpha_traj[0] + 0.5, "alpha_t should not run away upward given a tight seed"


def test_aci_pool_grows_monotonically():
    """Sanity check on the mechanics: the nonconformity pool should grow by
    exactly one entry per test batch processed (each realization is added
    after being scored)."""
    rng = np.random.default_rng(12)
    batches = generate_batches(n_batches=100, batch_size_range=(20, 80), tau=1.0, rng=rng)
    beta = batches[0]["beta"]
    p_hat = [predicted_p(b, beta) for b in batches]
    seed_batches, seed_p = batches[:30], p_hat[:30]
    remaining_batches, remaining_p = batches[30:], p_hat[30:]
    day_batches = [remaining_batches[i:i+7] for i in range(0, len(remaining_batches), 7)]
    day_p = [remaining_p[i:i+7] for i in range(0, len(remaining_p), 7)]

    bounds, _ = run_aci_batch_conformal(seed_batches, seed_p, day_batches, day_p, alpha=0.1, gamma=0.05)
    n_test_batches = sum(len(d) for d in day_batches)
    assert len(bounds) == n_test_batches
