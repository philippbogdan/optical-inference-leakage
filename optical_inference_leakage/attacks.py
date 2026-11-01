"""Simulate a client that measures all weight modes and fits a linear estimator. The reported MSE information statistic is a Monte Carlo estimate, not a certified finite-sample lower bound."""
import numpy as np

def p0_attack_all_modes(nu, P, K=1.0, S=64, seed=0, chunk=200000):
    """Draw Gaussian weights and averaged homodyne noise. Fit on half the samples and evaluate on the rest; return a plug-in information estimate and an approximate interval from variation across modes."""
    from .leakage import waterfill_capacity
    rs = np.random.default_rng(seed)
    nu = np.asarray(nu, float).ravel()
    _, p = waterfill_capacity(nu / K, P)
    act = np.nonzero(p > 0)[0]
    tot, var = (0.0, 0.0)
    for c0 in range(0, act.size, chunk):
        idx = act[c0:c0 + chunk]
        pv, nv = (p[idx], nu[idx] / K)
        w = rs.normal(size=(idx.size, S)) * np.sqrt(pv)[:, None]
        y = w + rs.normal(size=(idx.size, S)) * np.sqrt(nv)[:, None]
        h = S // 2
        beta = np.sum(w[:, :h] * y[:, :h], 1) / np.maximum(np.sum(y[:, :h] ** 2, 1), 1e-300)
        err = w[:, h:] - beta[:, None] * y[:, h:]
        mse = np.mean(err ** 2, 1)
        lb = 0.5 * np.log2(pv / mse)
        tot += lb.sum()
        var += np.var(lb) * idx.size
    se = np.sqrt(var)
    return (tot, (tot - 1.96 * se, tot + 1.96 * se))
