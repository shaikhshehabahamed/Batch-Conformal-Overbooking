"""Translate a calibrated upper-bound method into an actual overbooking
DECISION: given a candidate pool of M patients wanting an appointment and a
fixed capacity C, how many N <= M should be booked so that the (1-alpha)
upper bound on realized attendance among the first N stays within C? Then
measure REALIZED outcomes (using true Y) under that policy: overflow
events, utilization, idle slots -- closing the loop from a statistical
interval to an OR decision with real cost consequences.
"""
import numpy as np


def max_bookable(p_hat_pool, bound_fn, capacity):
    """p_hat_pool: predicted attend-probabilities for the M candidates, in
    booking-priority order (first-come-first-served, the order they'd
    actually be offered a slot). bound_fn(p_hat_subset) -> scalar upper
    bound. Returns the largest N in [0, M] with bound_fn(p_hat_pool[:N]) <=
    capacity.

    Full scan, not an early-break search: bound_fn(p_hat_pool[:n]) =
    mu(n) + q_hat*sigma(n) is non-decreasing in n whenever q_hat >= 0 (mu
    and sigma both grow with n since p_hat >= 0), which holds for every
    bound_fn actually used in this codebase at alpha < 0.5 -- but q_hat is
    an empirically-calibrated quantile with no built-in sign guarantee (an
    audit found real, non-monotone counterexamples on real data once alpha
    pushes q_hat negative, e.g. alpha >= 0.3 on the hotel dataset). An
    early-break-on-first-failure scan would silently under-book in that
    regime. A full scan costs the same O(M) bound_fn calls either way (M is
    at most a few hundred here) and is exact regardless of monotonicity.
    """
    M = len(p_hat_pool)
    if M == 0:
        return 0
    best_n = 0
    for n in range(1, M + 1):
        if bound_fn(p_hat_pool[:n]) <= capacity:
            best_n = n
    return best_n


def evaluate_policy(batches, p_hat_list, bound_fn, capacity):
    """For each (batch, p_hat) pair, book max_bookable patients, then look
    at the REAL outcomes among the booked patients. `capacity` may be a
    single scalar (same capacity for every batch) or a sequence of
    per-batch capacities (matching len(batches)) -- each record stores its
    OWN capacity so summarize_policy never has to guess which was used.
    Returns per-batch records: n_booked, realized_attendance, capacity,
    overflow (bool), utilization, idle_slots.
    """
    if np.isscalar(capacity):
        capacity = [capacity] * len(batches)
    records = []
    for batch, p_hat, cap in zip(batches, p_hat_list, capacity):
        n_booked = max_bookable(p_hat, bound_fn, cap)
        realized = int(np.sum(batch["Y"][:n_booked])) if n_booked > 0 else 0
        records.append({
            "n_booked": n_booked, "realized": realized, "capacity": cap,
            "overflow": realized > cap, "utilization": min(realized, cap) / cap,
            "idle_slots": max(0, cap - realized),
        })
    return records


def summarize_policy(records):
    overflow_amounts = [max(0, r["realized"] - r["capacity"]) for r in records]
    return {
        "overflow_rate": np.mean([r["overflow"] for r in records]),
        "mean_overflow_amount": np.mean(overflow_amounts),
        "mean_n_booked": np.mean([r["n_booked"] for r in records]),
        "mean_utilization": np.mean([r["utilization"] for r in records]),
        "mean_idle_slots": np.mean([r["idle_slots"] for r in records]),
    }
