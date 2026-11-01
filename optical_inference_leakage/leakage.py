"""Gaussian optical entropy and parallel-channel capacity in bits."""
import numpy as np

def g(nu):
    """Entropy of a Gaussian optical mode; the vacuum symplectic eigenvalue is one."""
    nu = np.asarray(nu, dtype=np.float64)
    a, b = ((nu + 1) / 2, (nu - 1) / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(b > 0, b * np.log2(np.where(b > 0, b, 1)), 0.0)
    return a * np.log2(a) - t

def broadcast_leak(gamma, K=1, nbar=None):
    """Holevo information per weight. With nbar supplied, gamma denotes the finite-brightness homodyne SNR. Without nbar, return the classical bright-limit expression."""
    gamma = np.asarray(gamma, dtype=np.float64)
    if nbar is None:
        return 0.5 * np.log2(1 + K * gamma)
    a2s = nbar
    a2v = (4 * a2s / gamma - 1) / 4
    if np.any(a2v < 0):
        raise ValueError('gamma above the coherent-state limit 4*nbar')
    nu_avg = np.sqrt((1 + 4 * a2v + 4 * K * a2s) * 1.0)
    nu_cond = np.sqrt((1 + 4 * a2v) * 1.0)
    return g(nu_avg) - g(nu_cond)

def waterfill_capacity(noise, power):
    """Distribute a fixed prior power across parallel Gaussian channels to maximise information."""
    nz = np.asarray(noise, dtype=np.float64)
    fin = np.isfinite(nz)
    n = np.sort(nz[fin])
    if n.size == 0 or power <= 0:
        return (0.0, np.zeros_like(nz))
    cs = np.cumsum(n)
    k = np.arange(1, n.size + 1)
    lev = (power + cs) / k
    ok = lev > n
    kk = np.max(np.nonzero(ok)[0]) + 1
    theta = (power + cs[kk - 1]) / kk
    p = np.where(fin, np.maximum(theta - np.where(fin, nz, 0), 0), 0)
    cap = 0.5 * np.sum(np.log2(1 + p[fin] / nz[fin]))
    return (float(cap), p)
