"""Compare a truncated Fock-space optical calculation with Gaussian product noise on a small instance."""
import numpy as np
from scipy.linalg import expm
from numpy.polynomial.hermite import hermgauss

def ladder(cut):
    return np.diag(np.sqrt(np.arange(1, cut)), 1)

def coherent(beta, cut):
    n = np.arange(cut)
    from scipy.special import gammaln
    c = np.exp(-abs(beta) ** 2 / 2 + n * np.log(abs(beta) + 1e-300) - 0.5 * gammaln(n + 1)) * np.sign(beta) ** n
    if beta == 0:
        c = np.zeros(cut)
        c[0] = 1.0
    return c

def givens_rotations(U):
    """Factor a real interferometer into pair rotations and phase signs."""
    n = U.shape[0]
    A = U.copy()
    ops = []
    for j in range(n):
        for i in range(n - 1, j, -1):
            a, b = (A[i - 1, j], A[i, j])
            if abs(b) < 1e-15:
                continue
            th = np.arctan2(b, a)
            G = np.eye(n)
            c, s = (np.cos(th), np.sin(th))
            G[[i - 1, i - 1, i, i], [i - 1, i, i - 1, i]] = [c, s, -s, c]
            A = G @ A
            ops.append((i - 1, i, th))
    return (ops, np.diag(A))

def apply_bs(psi, i, j, th, cut, n):
    """Apply a two-mode beamsplitter in the truncated tensor-product basis."""
    a = ladder(cut)
    I = np.eye(cut)

    def op(k, M):
        out = np.array([[1.0]])
        for q in range(n):
            out = np.kron(out, M if q == k else I)
        return out
    psi = psi.reshape([cut] * n)
    psi = np.moveaxis(psi, [i, j], [0, 1])
    sh = psi.shape
    two = psi.reshape(cut * cut, -1)
    A = np.kron(a, I)
    B = np.kron(I, a)
    H = th * (A.conj().T @ B - B.conj().T @ A)
    Ub = expm(-H)
    two = Ub @ two
    psi = two.reshape(sh)
    psi = np.moveaxis(psi, [0, 1], [i, j])
    return psi.reshape(-1)

def apply_signs(psi, D, cut, n):
    t = psi.reshape([cut] * n)
    par = (-1.0) ** np.arange(cut)
    for q in range(n):
        if D[q] < 0:
            sh = [1] * n
            sh[q] = cut
            t = t * par.reshape(sh)
    return t.reshape(-1)

def quad_moments(psi, modes, cut, n):
    """Calculate quadrature means and covariance from a state vector."""
    a = ladder(cut)
    X = a + a.T
    psi_t = psi.reshape([cut] * n)

    def apply(M, k, t):
        return np.moveaxis(np.tensordot(M, t, axes=([1], [k])), 0, k)
    mu = np.array([np.vdot(psi_t, apply(X, k, psi_t)).real for k in modes])
    C = np.zeros((len(modes), len(modes)))
    for p, k in enumerate(modes):
        for q, l in enumerate(modes):
            t = apply(X, l, psi_t)
            t = apply(X, k, t)
            C[p, q] = np.vdot(psi_t, t).real - mu[p] * mu[q]
    return (mu, C)

def exact_readout(w, X, alpha, v, cut=18, nq=5):
    """Integrate the modulated optical readout with Gaussian quadrature. Finite Fock truncation and quadrature order limit accuracy."""
    n, T = X.shape
    Q, _ = np.linalg.qr(X)
    r = Q.shape[1]
    full = np.linalg.qr(np.hstack([Q, np.random.RandomState(1).randn(n, n - r)]))[0]
    full[:, :r] = Q
    U = full.T
    ops, D = givens_rotations(U)
    coef = X.T @ Q
    gx, gw = hermgauss(nq)
    gx = gx * np.sqrt(2 * v)
    gw = gw / np.sqrt(np.pi)
    import itertools
    m1 = np.zeros(T)
    m2 = np.zeros((T, T))
    for idx in itertools.product(range(nq), repeat=n):
        zeta = gx[list(idx)]
        pw = np.prod(gw[list(idx)])
        psi = np.array([1.0])
        for j in range(n):
            psi = np.kron(psi, coherent(alpha * (w[j] + zeta[j]), cut))
        psi = apply_signs(psi, D, cut, n)
        for i, j, th in ops[::-1]:
            psi = apply_bs(psi, i, j, th, cut, n)
        mu, C = quad_moments(psi, list(range(r)), cut, n)
        ym = coef @ mu / (2 * alpha)
        yc = coef @ C @ coef.T / (4 * alpha ** 2)
        m1 += pw * ym
        m2 += pw * (yc + np.outer(ym, ym))
    return (m1, m2 - np.outer(m1, m1))

def model_readout(w, X, alpha, v):
    """Return product moments including modulation noise and vacuum noise."""
    return (X.T @ w, (v + 1 / (4 * alpha ** 2)) * (X.T @ X))
