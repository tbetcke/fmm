"""Double-precision cross-check of docs/CONVENTIONS.md (CONVENTION_VERSION = 1).

Independent second opinion next to the mpmath fixtures (T3). Checks, against SciPy's
Legendre functions with the Condon-Shortley phase removed:
  - Cartesian recursions of §3.5 vs the definitions of §3.3
  - separation identity and addition theorem of §3.4
  - gradient ladder of §3.4 (finite differences)
  - orthogonality of N D^n N^-1 for the rotation blocks of §3.8

Run: python3 tools/fixtures/crosscheck_scipy.py   (needs numpy, scipy)
Exits non-zero if any check exceeds its tolerance.
"""
import sys
from math import factorial as f

import numpy as np
from scipy.special import lpmv
from scipy.spatial.transform import Rotation

rng = np.random.default_rng(1)


def legendre(n, m, t):
    # SciPy includes the Condon-Shortley phase; §3.3 does not.
    return (-1) ** m * lpmv(m, n, t)


def spherical(x):
    r = np.linalg.norm(x)
    return r, np.arccos(x[2] / r), np.arctan2(x[1], x[0])


def R_def(n, m, x):
    r, th, ph = spherical(x)
    a = abs(m)
    v = r**n / f(n + a) * legendre(n, a, np.cos(th)) * np.exp(1j * a * ph)
    return v if m >= 0 else (-1) ** a * np.conj(v)


def I_def(n, m, x):
    r, th, ph = spherical(x)
    a = abs(m)
    v = f(n - a) / r ** (n + 1) * legendre(n, a, np.cos(th)) * np.exp(1j * a * ph)
    return v if m >= 0 else (-1) ** a * np.conj(v)


def R_rec(p, x):
    R = {(0, 0): 1.0 + 0j}
    rho, z, r2 = x[0] + 1j * x[1], x[2], x @ x
    for m in range(p + 1):
        if m > 0:
            R[(m, m)] = rho / (2 * m) * R[(m - 1, m - 1)]
        if m + 1 <= p:
            R[(m + 1, m)] = z * R[(m, m)]
        for n in range(m + 2, p + 1):
            R[(n, m)] = ((2 * n - 1) * z * R[(n - 1, m)] - r2 * R[(n - 2, m)]) / ((n + m) * (n - m))
    return R


def I_rec(p, x):
    rho, z, r2 = x[0] + 1j * x[1], x[2], x @ x
    I = {(0, 0): 1 / np.sqrt(r2) + 0j}
    for m in range(p + 1):
        if m > 0:
            I[(m, m)] = (2 * m - 1) * rho / r2 * I[(m - 1, m - 1)]
        if m + 1 <= p:
            I[(m + 1, m)] = (2 * m + 1) * z / r2 * I[(m, m)]
        for n in range(m + 2, p + 1):
            I[(n, m)] = ((2 * n - 1) * z * I[(n - 1, m)] - ((n - 1) ** 2 - m * m) * I[(n - 2, m)]) / r2
    return I


def real_vec(n, x):
    # Real storage of §3.6 for one degree: slots -n..n.
    v = np.zeros(2 * n + 1)
    for m in range(-n, n + 1):
        c = R_def(n, abs(m), x)
        v[n + m] = c.real if m >= 0 else c.imag
    return v


checks = []
p = 12
x = rng.normal(size=3)
R, I = R_rec(p, x), I_rec(p, x)
checks.append(("R recursion", max(abs(R[k] - R_def(*k, x)) for k in R), 1e-13))
checks.append(("I recursion (rel)", max(abs(I[k] - I_def(*k, x)) / abs(I_def(*k, x)) for k in I), 1e-13))

y = rng.normal(size=3) * 0.2
X = rng.normal(size=3)
X = X / np.linalg.norm(X) * 2
s = sum(np.conj(R_def(n, m, y)) * I_def(n, m, X) for n in range(40) for m in range(-n, n + 1))
checks.append(("separation identity", abs(s - 1 / np.linalg.norm(X - y)), 1e-13))

a, b = rng.normal(size=3), rng.normal(size=3)
err = 0.0
for n in range(7):
    for m in range(-n, n + 1):
        s = sum(
            R_def(k, l, a) * R_def(n - k, m - l, b)
            for k in range(n + 1)
            for l in range(-k, k + 1)
            if abs(m - l) <= n - k
        )
        err = max(err, abs(s - R_def(n, m, a + b)))
checks.append(("addition theorem", err, 1e-13))

h, err = 1e-6, 0.0
Rv = lambda nn, mm: R_def(nn, mm, x) if abs(mm) <= nn else 0
for n in range(1, 7):
    for m in range(-n, n + 1):
        g = [(R_def(n, m, x + h * e) - R_def(n, m, x - h * e)) / (2 * h) for e in np.eye(3)]
        err = max(err, abs(g[2] - Rv(n - 1, m)), abs(g[0] - 1j * g[1] - Rv(n - 1, m - 1)),
                  abs(g[0] + 1j * g[1] + Rv(n - 1, m + 1)))
checks.append(("gradient ladder (finite diff.)", err, 1e-8))

Q = Rotation.random(random_state=3).as_matrix()
for n in (3, 8):
    pts = rng.normal(size=(4 * n + 4, 3))
    A = np.array([real_vec(n, q) for q in pts])
    B = np.array([real_vec(n, Q @ q) for q in pts])
    D = np.linalg.lstsq(A, B, rcond=None)[0].T
    N = np.array([np.sqrt(f(n + abs(m)) * f(n - abs(m))) * (1 if m == 0 else np.sqrt(2)) for m in range(-n, n + 1)])
    W = np.diag(N) @ D @ np.diag(1 / N)
    checks.append((f"N D^{n} N^-1 orthogonal", np.abs(W @ W.T - np.eye(2 * n + 1)).max(), 1e-12))

ok = True
for name, e, tol in checks:
    status = "ok" if e <= tol else "FAIL"
    ok &= e <= tol
    print(f"{status:4}  {name:32} {e:.2e}  (tol {tol:.0e})")
sys.exit(0 if ok else 1)
