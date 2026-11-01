"""Conditional leakage floors from a supporting cut in noise variance. See THEORY.md for the statistical and convexity assumptions."""
import math
import numpy as np
from statistics import NormalDist
GRID = np.exp(np.linspace(-20, 20, 6001))

def bonferroni_z(L, alpha=0.05):
    """Normal quantile for simultaneous two-sided gradient intervals."""
    return NormalDist().inv_cdf(1 - alpha / (2 * L))

def cut_dual_floor(H_tabs, vstar, g_lo, g_hi, slack, weights=None, grid=GRID):
    """Bound the continuum infimum by interval endpoints, then maximise a valid dual value over non-negative multipliers. Information decreases with variance; the clipped cut increases."""
    n = len(H_tabs)
    vstar = np.asarray(vstar, float)
    g_lo = np.maximum(np.asarray(g_lo, float), 0.0)
    g_hi = np.maximum(np.asarray(g_hi, float), g_lo)
    weights = [np.ones(n)] if weights is None else weights
    d = vstar[:, None] * (grid[None, :] - 1.0)
    f = np.where(d >= 0, g_lo[:, None] * d, g_hi[:, None] * d)
    f0 = -g_hi * vstar
    H = np.array([np.asarray(h, float) for h in H_tabs])
    scale = max(float(np.sum(g_hi * vstar)), 1e-300)
    best_all = 0.0
    for w in weights:
        w = np.asarray(w, float)
        wH = w[:, None] * H

        def dual(mu):
            inner = np.minimum.reduce([wH[:, 0] + mu * f0, (wH[:, 1:] + mu * f[:, :-1]).min(1), mu * f[:, -1]])
            return float(inner.sum() - mu * slack)
        mus = np.exp(np.linspace(-80, 80, 641)) / scale
        vals = np.array([dual(m) for m in mus])
        j = int(np.argmax(vals))
        best = vals[j]
        lo, hi = (mus[max(j - 1, 0)], mus[min(j + 1, len(mus) - 1)])
        for _ in range(50):
            m2 = np.exp(np.linspace(math.log(lo), math.log(hi), 9))
            v2 = np.array([dual(m) for m in m2])
            j2 = int(np.argmax(v2))
            best = max(best, v2[j2])
            lo, hi = (m2[max(j2 - 1, 0)], m2[min(j2 + 1, 8)])
        best_all = max(best_all, best)
    return max(best_all, 0.0)

def capacity_tab(N, s, vstar, K, grid=GRID):
    """Total Gaussian information for K independent copies, evaluated on a variance grid."""
    return N / 2 * np.log2(1 + K * s / (vstar * grid))
