import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batch_risk_control.decision import max_bookable, evaluate_policy, summarize_policy


def test_max_bookable_hand_case():
    """4 candidates, p_hat=[0.5]*4. Bound function: just the running sum
    (mu-only, no uncertainty term, for a clean hand check). Capacity=2.1
    means at most 4 candidates fit (sum of 4 * 0.5 = 2.0 <= 2.1), a 5th
    would push it over."""
    p_hat_pool = np.array([0.5, 0.5, 0.5, 0.5])
    bound_fn = lambda p: np.sum(p)
    n = max_bookable(p_hat_pool, bound_fn, capacity=2.1)
    assert n == 4
    n2 = max_bookable(p_hat_pool, bound_fn, capacity=1.9)
    assert n2 == 3  # sum of 3 = 1.5 <= 1.9, sum of 4 = 2.0 > 1.9


def test_max_bookable_zero_capacity():
    p_hat_pool = np.array([0.1, 0.1])
    bound_fn = lambda p: np.sum(p)
    assert max_bookable(p_hat_pool, bound_fn, capacity=0.0) == 0


def test_max_bookable_empty_pool():
    bound_fn = lambda p: np.sum(p)
    assert max_bookable(np.array([]), bound_fn, capacity=10) == 0


def test_evaluate_policy_realized_matches_actual_outcomes():
    """If we book all 3 candidates (bound trivially satisfied) the realized
    attendance must equal the true sum of Y among booked patients."""
    batches = [{"Y": np.array([1, 0, 1])}]
    p_hat_list = [np.array([0.9, 0.9, 0.9])]
    bound_fn = lambda p: 1000  # always satisfied -> books everyone
    records = evaluate_policy(batches, p_hat_list, bound_fn, capacity=1000)
    assert records[0]["n_booked"] == 3
    assert records[0]["realized"] == 2  # 1+0+1


def test_tighter_bound_books_more_at_same_capacity():
    """A tighter (smaller-multiplier) bound function should book at least
    as many candidates as a looser one, for the same capacity -- this is
    the direct mechanism by which interval tightness becomes utilization."""
    rng = np.random.default_rng(0)
    p_hat_pool = rng.uniform(0.6, 0.9, size=30)
    tight_bound = lambda p: np.sum(p) + 1.0 * np.sqrt(np.sum(p * (1 - p)))
    loose_bound = lambda p: np.sum(p) + 3.0 * np.sqrt(np.sum(p * (1 - p)))
    capacity = 20.0
    n_tight = max_bookable(p_hat_pool, tight_bound, capacity)
    n_loose = max_bookable(p_hat_pool, loose_bound, capacity)
    assert n_tight >= n_loose


def test_summarize_policy_overflow_rate_in_unit_interval():
    batches = [{"Y": np.array([1, 1, 0])}, {"Y": np.array([0, 0, 1])}]
    p_hat_list = [np.array([0.9, 0.9, 0.9]), np.array([0.3, 0.3, 0.3])]
    bound_fn = lambda p: np.sum(p) + 0.5
    records = evaluate_policy(batches, p_hat_list, bound_fn, capacity=1.5)
    summary = summarize_policy(records)
    assert 0 <= summary["overflow_rate"] <= 1
    assert summary["mean_utilization"] >= 0


def test_max_bookable_correct_under_non_monotone_bound():
    """Regression test for an audit-found bug: bound_fn(p[:n]) = mu(n) +
    q*sigma(n) is only guaranteed non-decreasing in n when q >= 0. With a
    NEGATIVE multiplier (a real, reproducible case on this project's own
    real data at large alpha, e.g. alpha=0.5 on the hotel dataset), the
    bound can dip non-monotonically as n grows -- an early-break scan would
    silently stop at the first dip and under-book, even though a larger n
    further along is also feasible. A full scan must find the true max."""
    p_hat_pool = np.full(30, 0.5)  # identical patients -> mu(n)=0.5n, sigma(n)=sqrt(0.25n)
    q_negative = -5.0  # verified below (not hand-derived) to produce real dips for integer n

    def bound_fn(p):
        mu = np.sum(p)
        sigma = np.sqrt(np.sum(p * (1 - p)))
        return mu + q_negative * sigma

    vals = [bound_fn(p_hat_pool[:n]) for n in range(1, 31)]
    assert any(vals[i + 1] < vals[i] for i in range(len(vals) - 1)), \
        "test setup should produce a non-monotone bound -- if this fails, the example no longer exercises the bug"

    # true max feasible n via brute force (ground truth) -- pick a capacity
    # that is reachable EARLY (before the dip) so an early-break scan would
    # stop there, while a larger n further along (after the dip recovers)
    # is also legitimately feasible.
    capacity = vals[0]  # n=1's bound; the dip means later n values also satisfy it
    true_max_n = max(n for n in range(1, 31) if bound_fn(p_hat_pool[:n]) <= capacity)
    assert true_max_n > 1, "test setup should have a feasible n beyond the first dip"

    assert max_bookable(p_hat_pool, bound_fn, capacity) == true_max_n


def test_evaluate_policy_accepts_per_batch_capacity_list():
    """Per-batch capacities must be respected individually, not averaged --
    a record's own overflow/utilization/idle_slots must reflect ITS
    capacity, not some pooled scalar."""
    batches = [{"Y": np.array([1, 1, 1])}, {"Y": np.array([1, 1, 1])}]
    p_hat_list = [np.array([0.9, 0.9, 0.9]), np.array([0.9, 0.9, 0.9])]
    bound_fn = lambda p: 0  # trivially satisfied for any nonnegative capacity -> books everyone
    records = evaluate_policy(batches, p_hat_list, bound_fn, capacity=[1, 5])
    assert records[0]["capacity"] == 1 and records[0]["overflow"] is True  # realized=3 > cap=1
    assert records[1]["capacity"] == 5 and records[1]["overflow"] is False  # realized=3 <= cap=5
    summary = summarize_policy(records)
    assert summary["overflow_rate"] == 0.5
