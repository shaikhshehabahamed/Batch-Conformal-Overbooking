"""Adaptive Conformal Inference (Gibbs & Candes 2021, arXiv:2106.00170)
applied to batch-level risk control: maintain a time-varying alpha_t,
updated online after each realization based on whether the previous bound
covered. Designed specifically for distribution drift over time -- exactly
the failure mode found in exp04/exp05 (static and rolling-window
calibration both left a real coverage gap under the dataset's genuine
temporal attendance-rate drift).

Update rule: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
  err_t = 1 if S_t > bound_t (missed), 0 if covered.
If recent batches have been missing (err_t=1 often), alpha_t DECREASES,
widening future bounds; if over-covering, alpha_t increases, tightening
them. gamma controls responsiveness.
"""
import numpy as np
from .methods import _mu_sigma


def run_aci_batch_conformal(cal_pool_batches, cal_pool_p_hat, test_batches_by_day,
                             test_p_hat_by_day, alpha=0.10, gamma=0.05):
    """cal_pool_batches/p_hat: the historical pool to draw nonconformity
    scores from (grows as we walk forward -- test batches are appended to
    it after being scored, so later days benefit from more history).
    test_batches_by_day / test_p_hat_by_day: lists of (batches, p_hat) per
    day, in chronological order.

    Returns per-batch bounds (flat list, same order as test_batches_by_day
    flattened) and the alpha_t trajectory (one value per day).
    """
    alpha_t = alpha
    alpha_trajectory = []
    all_bounds = []

    # running pool of nonconformity scores R = (S_b - mu_hat_b)/sigma_hat_b
    R_pool = []
    for batch, p_hat in zip(cal_pool_batches, cal_pool_p_hat):
        mu, sigma = _mu_sigma(p_hat)
        R_pool.append((np.sum(batch["Y"]) - mu) / sigma)
    R_pool = list(R_pool)

    for day_batches, day_p_hat in zip(test_batches_by_day, test_p_hat_by_day):
        alpha_trajectory.append(alpha_t)
        a_eff = min(max(alpha_t, 1e-3), 0.999)
        n = len(R_pool)
        k = int(np.ceil((n + 1) * (1 - a_eff)))
        k = min(max(k, 1), n)
        q_hat = np.sort(R_pool)[k - 1]

        day_bounds = []
        for batch, p_hat in zip(day_batches, day_p_hat):
            mu, sigma = _mu_sigma(p_hat)
            bound = mu + q_hat * sigma
            day_bounds.append(bound)
            S_b = np.sum(batch["Y"])
            err = 1.0 if S_b > bound else 0.0
            alpha_t = alpha_t + gamma * (alpha - err)
            R_pool.append((S_b - mu) / sigma)  # grow the pool with realized data

        all_bounds.extend(day_bounds)

    return np.array(all_bounds), alpha_trajectory
