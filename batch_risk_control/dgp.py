"""Monte Carlo DGP for batch attendance with a KNOWN within-batch
correlation structure (a shared batch-level random effect), so we can check
whether a candidate method's coverage of S_b (observable, real target) is
actually valid under REAL correlation the classical independent-Bernoulli
approximation ignores.
"""
import numpy as np


def generate_batches(n_batches, batch_size_range=(20, 80), n_features=3,
                      beta=None, tau=1.0, rng=None):
    """Generate n_batches independent batches, each with a random size,
    patient covariates, a shared batch-level logit shift (tau controls its
    strength -> induces within-batch correlation), and realized outcomes.

    Returns a list of dicts, one per batch:
      X: (n_i, n_features), true_p: (n_i,), Y: (n_i,) in {0,1}, eta: scalar
    """
    if rng is None:
        rng = np.random.default_rng()
    if beta is None:
        beta = rng.normal(0, 1, size=n_features)

    batches = []
    for b in range(n_batches):
        n_i = rng.integers(batch_size_range[0], batch_size_range[1] + 1)
        X = rng.normal(0, 1, size=(n_i, n_features))
        eta = rng.normal(0, tau)  # shared batch-level effect (day/clinic factor)
        logit_attend = X @ beta + eta
        true_p = 1 / (1 + np.exp(-logit_attend))  # true P(attend)
        Y = rng.binomial(1, true_p)
        batches.append({"X": X, "true_p": true_p, "Y": Y, "eta": eta, "beta": beta})
    return batches


def predicted_p(batch, beta_hat):
    """Classifier prediction WITHOUT access to the batch effect eta -- a
    realistic classifier fit on covariates alone can't see an unmeasured
    shared factor. This is what creates the correlation gap the naive
    parametric approximation misses.
    """
    logit = batch["X"] @ beta_hat
    return 1 / (1 + np.exp(-logit))
