"""Two-index (alternating h/T) Monte Carlo estimator for homogenized coefficients, d = 2.

Implements the scheme of B. C. White, "Two-Index Monte Carlo Method for Approximating
Coefficients of Linear Elliptic Problems" (ISTA MSc thesis, 2026):

  * coefficient field   a(x) = A(g(x)) I,  A(g) = exp(beta * tanh g),
                        g a centred, stationary, smooth Gaussian field on the torus
                        U_L = [0, L)^2 (Definition 12, "tame Gaussian coefficients");
  * full corrector      phi in V_h:  int a (e_i + grad phi).grad eta + (1/T) phi eta = 0
                        (Definition 9), V_h = periodic Lagrange elements with the standard
                        nodal basis on a uniform n x n mesh: either P_p on triangles (each
                        square split along its (1,1) diagonal; the elements the thesis
                        analyses, Config.element = "P") or Q_p on quadrilaterals ("Q");
  * coefficient         a^{L,T,h} e_i = L^{-2} int_U a (e_i + grad phi_i)   (Definition 10.3),
                        evaluated with Gauss quadrature exact for degree 2p+1 ((p+1)^2
                        points per quadrilateral or per triangle);
  * estimator           a_MC of Definition 13 / Eq. (5) with sample sizes of Eq. (11);
  * linear solver       unpreconditioned conjugate gradients (as assumed in Lemma 15).

Theorem numbers follow the final version of the thesis (Theorem 17 needs p >= d + 1 = 3).

Reference value: for this ensemble the law of a equals the law of 1/a (g -> -g) and is
invariant under 90-degree rotations, so by Dykhne's duality the exact homogenized
matrix is  a_bar = I.  Errors are measured against I.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    p: int = 2                 # polynomial degree of the Q_p elements (scheme parameter)
    nu: float = 3.5            # Matern smoothness; Def. 12 needs nu > p + 1
    ell: float = 0.25          # Matern length parameter (correlation ~0.29 at |x| = 1)
    beta: float = 1.0          # contrast: a in (e^-beta, e^beta)
    tail_tol: float = 1e-6     # discarded fraction of the field variance (Fourier cut-off)
    cg_c: float = 1e-4         # CG relative tolerance = cg_c * (h/L)^d   (Lemma 15: eps ~ (h/L)^d)
    cg_maxiter: int = 200_000
    element: str = "Q"         # "P": P_p Lagrange on triangles;  "Q": Q_p Lagrange on quadrilaterals
    kmax: float = math.inf     # optional absolute Fourier cut-off (used only for rough fields, small nu)
    aniso: float = 1.0         # anisotropy: a(x) = s(M^{-1/2} x) M with M = R(theta) diag(sqrt(aniso), 1/sqrt(aniso)) R(theta)^T
    theta: float = 0.0         # rotation angle of M in degrees; the exact homogenized coefficient is then M

    def to_dict(self):
        return asdict(self)


def aniso_matrix(cfg: Config):
    """The constant matrix M of the anisotropic coefficient a(x) = s(M^{-1/2} x) M, or None
    for the isotropic case (aniso = 1).

    With s the isotropic scalar field, the substitution x = M^{1/2} y turns the corrector problem
    for a into the isotropic one for s, so ā = M^{1/2} ā_s M^{1/2} = M exactly (ā_s = I by
    duality). det M = 1, so the field keeps unit variance and the same spectral normalisation.
    """
    if cfg.aniso == 1.0:
        return None
    c, s = math.cos(math.radians(cfg.theta)), math.sin(math.radians(cfg.theta))
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([math.sqrt(cfg.aniso), 1.0 / math.sqrt(cfg.aniso)]) @ R.T


def reference_abar(cfg: Config) -> np.ndarray:
    """Exact homogenized coefficient of the ensemble: I (Keller-Dykhne), or M if anisotropic."""
    M = aniso_matrix(cfg)
    return np.eye(2) if M is None else M


# --------------------------------------------------------------------------------------
# Gaussian coefficient field on the torus
# --------------------------------------------------------------------------------------
class GaussianCoefficient:
    """Periodised Matern-type Gaussian field on [0, L)^2, unit variance.

    g(x) = sum_m ghat_m exp(i k_m . x),   k_m = 2 pi m / L,
    E|ghat_m|^2 = S(k_m) (2 pi / L)^2 / int S,   S(k) = (1 + ell^2 |k|^2)^(-(nu + 1)).

    Sampling the spectral density on the dual lattice is exactly periodising the
    whole-space covariance, so the ensemble is the same for every L.  S(k) decays like
    |k|^(-2 nu - 2) = |k|^(-d - 2(p+1) - eps) for nu > p + 1, as Definition 12 requires.
    Modes with |k| > K are dropped, where (ell K)^(-2 nu) = tail_tol is the discarded
    variance fraction; the truncated ensemble keeps all symmetries (a <-> 1/a, rotations).
    """

    def __init__(self, L: float, cfg: Config):
        self.L, self.cfg = L, cfg
        kcut = min(cfg.tail_tol ** (-1.0 / (2 * cfg.nu)) / cfg.ell, cfg.kmax)
        M = aniso_matrix(cfg)
        # anisotropic: covariance C_iso(M^{-1/2} x) has spectral density S_iso(k . M k) (det M = 1);
        # the ellipse k . M k <= kcut^2 reaches out to |k| = kcut / sqrt(lambda_min(M))
        kbox = kcut if M is None else kcut / math.sqrt(np.linalg.eigvalsh(M)[0])
        mmax = int(math.ceil(kbox * L / (2 * math.pi)))
        self.Mg = Mg = 2 * mmax + 1                       # odd => exact Hermitian symmetry
        self.m = np.fft.fftfreq(Mg, 1.0 / Mg).round().astype(np.int64)
        k = 2 * math.pi * self.m / L
        if M is None:
            k2 = k[:, None] ** 2 + k[None, :] ** 2
        else:
            k2 = M[0, 0] * k[:, None] ** 2 + 2 * M[0, 1] * k[:, None] * k[None, :] + M[1, 1] * k[None, :] ** 2
        S = (1.0 + cfg.ell ** 2 * k2) ** (-(cfg.nu + 1.0))
        S[k2 > kcut ** 2] = 0.0
        S_int = math.pi / (cfg.ell ** 2 * cfg.nu)       # int_{R^2} S(k) dk
        # fft2 of real white noise has E|.|^2 = Mg^2 per mode
        self.amp = np.sqrt(S / S_int) * (2 * math.pi / L) / Mg
        self.variance = float((self.amp ** 2).sum() * Mg ** 2)
        self._fold_cache: dict = {}

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Fourier coefficients ghat (Mg x Mg, FFT ordering) of one realisation."""
        return np.fft.fft2(rng.standard_normal((self.Mg, self.Mg))) * self.amp

    def _fold(self, n: int, shift: float) -> sp.csr_matrix:
        """Sparse n x Mg map  c -> sum_{m = j mod n} c_m exp(2 pi i m shift / n)."""
        key = (n, shift)
        F = self._fold_cache.get(key)
        if F is None:
            phase = np.exp(2j * math.pi * self.m * shift / n)
            F = sp.csr_matrix((phase, (self.m % n, np.arange(self.Mg))), shape=(n, self.Mg))
            self._fold_cache[key] = F
        return F

    def g_on_grid(self, ghat: np.ndarray, n: int, shifts: np.ndarray) -> np.ndarray:
        """g at the points x = (L/n)(i + s_x, j + s_y), s_x, s_y in `shifts`.

        Exact evaluation of the trigonometric polynomial: for a fixed sub-cell shift the
        points form a uniform n-grid, so after folding aliased modes (m mod n) and applying
        the shift phase it is a plain inverse FFT of size n x n.
        Returns an array of shape (n, nq, n, nq) indexed [ix, qx, iy, qy].
        """
        nq = len(shifts)
        out = np.empty((n, nq, n, nq))
        for qx, sx in enumerate(shifts):
            B = self._fold(n, sx) @ ghat                       # (n, Mg)
            for qy, sy in enumerate(shifts):
                C = (self._fold(n, sy) @ B.T).T                # (n, n)
                out[:, qx, :, qy] = (np.fft.ifft2(C) * (n * n)).real
        return out

    def g_at_offsets(self, ghat: np.ndarray, n: int, offsets: np.ndarray) -> np.ndarray:
        """g at x = (L/n)(i + s_1, j + s_2) for every cell (i, j) and every offset (s_1, s_2).

        Same exact folding/FFT evaluation as g_on_grid, for offsets that are not a tensor
        grid (triangle quadrature). Returns shape (n, n, len(offsets)) indexed [ix, iy, q].
        """
        out = np.empty((n, n, len(offsets)))
        sx_unique = np.unique(offsets[:, 0])
        for sx in sx_unique:
            B = self._fold(n, float(sx)) @ ghat                # (n, Mg)
            for q in np.nonzero(offsets[:, 0] == sx)[0]:
                C = (self._fold(n, float(offsets[q, 1])) @ B.T).T
                out[:, :, q] = (np.fft.ifft2(C) * (n * n)).real
        return out

    def coefficient(self, g: np.ndarray) -> np.ndarray:
        return np.exp(self.cfg.beta * np.tanh(g))


# --------------------------------------------------------------------------------------
# Periodic Q_p finite elements on a uniform quadrilateral mesh
# --------------------------------------------------------------------------------------
def lagrange_1d(p: int, x: np.ndarray):
    """Values and derivatives of the equispaced-node Lagrange basis on [0, 1] at x."""
    nodes = np.linspace(0.0, 1.0, p + 1)
    C = np.linalg.inv(np.vander(nodes, p + 1, increasing=True))   # coeffs (power, basis)
    V = np.vander(x, p + 1, increasing=True)
    dV = np.zeros_like(V)
    dV[:, 1:] = V[:, :-1] * np.arange(1, p + 1)
    return V @ C, dV @ C


class QpMesh:
    """Q_p Lagrange space on the periodic n x n mesh of [0, L)^2 (h = L / n).

    DOFs: nodes (i/p) h on each axis, i = 0..n p - 1, periodic; Ndof = (n p)^2.
    Quadrature: tensor Gauss-Legendre with p + 1 points per direction (exact for Q_{2p+1}).
    """

    def __init__(self, L: float, n: int, p: int, M=None):
        self.L, self.n, self.p = L, n, p
        self.M = M                                    # constant matrix factor of a = s M (None: identity)
        self.h = h = L / n
        self.Np = Np = n * p
        self.Ndof = Np * Np
        nq = p + 1
        xg, wg = leggauss(nq)
        self.xq = 0.5 * (xg + 1.0)                    # Gauss points on [0, 1]
        wq = 0.5 * wg
        B, D = lagrange_1d(p, self.xq)                # (nq, p+1)
        nloc = (p + 1) ** 2
        # reference-element tables, rows Q = qx*nq + qy, cols A = a*(p+1) + b
        N2 = np.einsum("xa,yb->xyab", B, B).reshape(nq * nq, nloc)
        Gx = np.einsum("xa,yb->xyab", D, B).reshape(nq * nq, nloc)
        Gy = np.einsum("xa,yb->xyab", B, D).reshape(nq * nq, nloc)
        w2 = np.outer(wq, wq).ravel()
        self.w2 = w2
        # in 2D the reference stiffness kernel is h-independent: (1/h)^2 * h^2
        self.Kker = _stiff_kernel(w2, Gx, Gy, M).reshape(nq * nq, nloc * nloc)
        Mloc = h * h * np.einsum("q,qa,qb->ab", w2, N2, N2)
        # load-vector kernels:  -int a d_i N_A = -h * sum_Q a_Q w_Q dN_A/dxi_i
        self.Rker = _load_kernels(h, w2, Gx, Gy, M)
        # connectivity
        e = np.arange(n)
        a = np.arange(p + 1)
        gx = (e[:, None] * p + a[None, :]) % Np                   # (n, p+1)
        conn = (gx[:, None, :, None] * Np + gx[None, :, None, :])  # (ex, ey, a, b)
        self.conn = conn.reshape(n * n, nloc)
        # CSR pattern and scatter map (element entry -> CSR data slot)
        rows = np.repeat(self.conn, nloc, axis=1).ravel()
        cols = np.tile(self.conn, (1, nloc)).ravel()
        keys = rows.astype(np.int64) * self.Ndof + cols
        ukeys, inv = np.unique(keys, return_inverse=True)
        self.scatter = inv.astype(np.int32)
        self.nnz = len(ukeys)
        urows = (ukeys // self.Ndof).astype(np.int64)
        self.indices = (ukeys % self.Ndof).astype(np.int32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(urows, minlength=self.Ndof))]).astype(np.int32)
        self.Mdata = np.bincount(self.scatter, weights=np.tile(Mloc.ravel(), n * n), minlength=self.nnz)

    def coef_at_quad(self, field: GaussianCoefficient, ghat: np.ndarray) -> np.ndarray:
        """a at the quadrature points, shape (n_el, nq^2)."""
        g = field.g_on_grid(ghat, self.n, self.xq)          # (ex, qx, ey, qy)
        nq = self.p + 1
        a4 = field.coefficient(g).transpose(0, 2, 1, 3)       # (ex, ey, qx, qy)
        return np.ascontiguousarray(a4).reshape(self.n * self.n, nq * nq)

    def stiffness_data(self, aq: np.ndarray) -> np.ndarray:
        return np.bincount(self.scatter, weights=(aq @ self.Kker).ravel(), minlength=self.nnz)

    def matrix(self, Kdata: np.ndarray, T: float) -> sp.csr_matrix:
        return sp.csr_matrix((Kdata + self.Mdata / T, self.indices, self.indptr),
                             shape=(self.Ndof, self.Ndof))

    def rhs(self, aq: np.ndarray) -> np.ndarray:
        """Columns b_i(eta) = -int a e_i . grad eta, i = 1, 2."""
        return np.stack([np.bincount(self.conn.ravel(), weights=(aq @ R).ravel(), minlength=self.Ndof)
                         for R in self.Rker], axis=1)

    def homogenized(self, aq: np.ndarray, b: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """a^{L,T,h}_{ji} = e_j . L^{-2} int a (e_i + grad phi_i)
                         = L^{-2} ( delta_ij int a  -  b_j . phi_i )   (same quadrature)."""
        mean_a = self.h * self.h * float((aq * self.w2).sum())
        return (mean_a * (np.eye(2) if self.M is None else self.M) - b.T @ phi) / self.L ** 2

    def quad_points(self):
        """Physical coordinates and weights of all quadrature points, each of shape (n_el * nq^2,)."""
        n, nq = self.n, self.p + 1
        x = self.h * (np.arange(n)[:, None] + self.xq[None, :])                 # (n, nq)
        X1 = np.broadcast_to(x[:, None, :, None], (n, n, nq, nq)).ravel()
        X2 = np.broadcast_to(x[None, :, None, :], (n, n, nq, nq)).ravel()
        W = np.broadcast_to(self.h ** 2 * self.w2, (n * n, nq * nq)).ravel()
        return X1, X2, W

    def eval_grad(self, phi: np.ndarray, x1: np.ndarray, x2: np.ndarray, chunk: int = 200_000):
        """Gradients of the FE functions phi (Ndof, m) at arbitrary points; returns (P, m, 2)."""
        p, n, h = self.p, self.n, self.h
        out = np.empty((len(x1), phi.shape[1], 2))
        for s in range(0, len(x1), chunk):
            u, v = x1[s:s + chunk] / h, x2[s:s + chunk] / h
            ex, ey = np.floor(u), np.floor(v)
            Bx, Dx = lagrange_1d(p, u - ex)
            By, Dy = lagrange_1d(p, v - ey)
            e = (ex.astype(np.int64) % n) * n + (ey.astype(np.int64) % n)
            coef = phi[self.conn[e]]                                            # (P, nloc, m)
            gx = np.einsum("Pa,Pb,Pabm->Pm", Dx, By, coef.reshape(len(e), p + 1, p + 1, -1)) / h
            gy = np.einsum("Pa,Pb,Pabm->Pm", Bx, Dy, coef.reshape(len(e), p + 1, p + 1, -1)) / h
            out[s:s + chunk, :, 0], out[s:s + chunk, :, 1] = gx, gy
        return out


# --------------------------------------------------------------------------------------
# Periodic P_p finite elements on a uniform triangulation (each square split in two)
# --------------------------------------------------------------------------------------
def triangle_rule(m: int):
    """Collapsed Gauss rule with m^2 points on the reference triangle {xi, eta >= 0, xi + eta <= 1},
    exact for polynomials of total degree 2m - 1 (Gauss-Jacobi(1, 0) in eta, Gauss-Legendre in t,
    xi = t (1 - eta)). Returns xi, eta, w with sum(w) = 1/2."""
    from scipy.special import roots_jacobi
    xs, ws = roots_jacobi(m, 1.0, 0.0)
    eta, weta = 0.5 * (xs + 1.0), ws / 4.0
    xt, wt = leggauss(m)
    t, wt = 0.5 * (xt + 1.0), 0.5 * wt
    E, Tt = np.meshgrid(eta, t, indexing="ij")
    W = np.outer(weta, wt)
    return (Tt * (1.0 - E)).ravel(), E.ravel(), W.ravel()


def _monomials(p):
    return [(i, j) for i in range(p + 1) for j in range(p + 1 - i)]


def _mono_eval(p, x, y):
    """Values and x-, y-derivatives of the monomials x^i y^j (i + j <= p) at points; each (P, nmono)."""
    mons = _monomials(p)
    V = np.stack([x ** i * y ** j for i, j in mons], axis=1)
    Dx = np.stack([i * x ** max(i - 1, 0) * y ** j for i, j in mons], axis=1)
    Dy = np.stack([j * x ** i * y ** max(j - 1, 0) for i, j in mons], axis=1)
    return V, Dx, Dy


class PpMesh:
    """P_p Lagrange space on the periodic triangulation of [0, L)^2 (h = L / n).

    Each of the n x n squares is split along its (1, 1) diagonal into a lower triangle
    (type 0: y <= x in cell coordinates) and an upper triangle (type 1: y >= x). The
    equispaced Lagrange nodes of both triangles together are the (p+1)^2 lattice points of the
    square, so the DOFs are again the periodic (n p) x (n p) lattice: Ndof = (n p)^2.
    Quadrature: collapsed Gauss rule with (p+1)^2 points per triangle, exact for degree 2p+1.
    Elements are ordered e = type * n^2 + ex * n + ey; arrays over quadrature points have
    shape (2 n^2, (p+1)^2) in that order.
    """

    def __init__(self, L: float, n: int, p: int, M=None):
        self.L, self.n, self.p = L, n, p
        self.M = M                                          # constant matrix factor of a = s M (None: identity)
        self.h = h = L / n
        self.Np = Np = n * p
        self.Ndof = Np * Np
        m = p + 1
        xi, eta, w = triangle_rule(m)
        self.w2 = w                                         # same weights for both types (|det J| = 1)
        self.nq = nq = len(w)
        mons = _monomials(p)
        self.nloc = nloc = len(mons)
        self.types = []
        offsets = []
        for typ in (0, 1):
            nodes = [(a, b) for a in range(p + 1) for b in range(p + 1) if (b <= a if typ == 0 else b >= a)]
            V, _, _ = _mono_eval(p, np.array([a for a, _ in nodes], float) / p,
                                 np.array([b for _, b in nodes], float) / p)
            C = np.linalg.inv(V)                              # (nmono, nloc): basis = mono @ C
            # reference triangle -> cell coordinates (|det| = 1)
            x, y = (xi + eta, eta) if typ == 0 else (xi, xi + eta)
            Vq, Dxq, Dyq = _mono_eval(p, x, y)
            N, Gx, Gy = Vq @ C, Dxq @ C, Dyq @ C              # (nq, nloc), cell-coordinate derivatives
            Kker = _stiff_kernel(w, Gx, Gy, M).reshape(nq, nloc * nloc)
            Mloc = h * h * np.einsum("q,qa,qb->ab", w, N, N)
            Rker = _load_kernels(h, w, Gx, Gy, M)
            e = np.arange(n)
            gx = (e[:, None, None] * p + np.array([a for a, _ in nodes])[None, None, :]) % Np
            gy = (e[None, :, None] * p + np.array([b for _, b in nodes])[None, None, :]) % Np
            conn = (gx * Np + gy).reshape(n * n, nloc)       # cell (ex, ey) -> row ex * n + ey
            self.types.append(dict(C=C, Kker=Kker, Mloc=Mloc, Rker=Rker, conn=conn))
            offsets.append(np.stack([x, y], axis=1))
        self.offsets = np.concatenate(offsets)               # (2 nq, 2), type-major
        self.conn = np.concatenate([t["conn"] for t in self.types])
        rows = np.repeat(self.conn, nloc, axis=1).ravel()
        cols = np.tile(self.conn, (1, nloc)).ravel()
        keys = rows.astype(np.int64) * self.Ndof + cols
        ukeys, inv = np.unique(keys, return_inverse=True)
        self.scatter = inv.astype(np.int32)
        self.nnz = len(ukeys)
        urows = (ukeys // self.Ndof).astype(np.int64)
        self.indices = (ukeys % self.Ndof).astype(np.int32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(urows, minlength=self.Ndof))]).astype(np.int32)
        Mel = np.concatenate([np.tile(t["Mloc"].ravel(), n * n) for t in self.types])
        self.Mdata = np.bincount(self.scatter, weights=Mel, minlength=self.nnz)

    def coef_at_quad(self, field: GaussianCoefficient, ghat: np.ndarray) -> np.ndarray:
        """a at the quadrature points, shape (2 n^2, nq)."""
        n, nq = self.n, self.nq
        g = field.g_at_offsets(ghat, n, self.offsets)        # (ex, ey, 2 nq)
        a = field.coefficient(g).reshape(n * n, 2, nq).transpose(1, 0, 2)
        return np.ascontiguousarray(a).reshape(2 * n * n, nq)

    def _split(self, aq):
        nn = self.n * self.n
        return aq[:nn], aq[nn:]

    def stiffness_data(self, aq: np.ndarray) -> np.ndarray:
        loc = np.concatenate([a @ t["Kker"] for a, t in zip(self._split(aq), self.types)])
        return np.bincount(self.scatter, weights=loc.ravel(), minlength=self.nnz)

    def matrix(self, Kdata: np.ndarray, T: float) -> sp.csr_matrix:
        return sp.csr_matrix((Kdata + self.Mdata / T, self.indices, self.indptr),
                             shape=(self.Ndof, self.Ndof))

    def rhs(self, aq: np.ndarray) -> np.ndarray:
        cols = []
        for i in range(2):
            loc = np.concatenate([a @ t["Rker"][i] for a, t in zip(self._split(aq), self.types)])
            cols.append(np.bincount(self.conn.ravel(), weights=loc.ravel(), minlength=self.Ndof))
        return np.stack(cols, axis=1)

    def homogenized(self, aq: np.ndarray, b: np.ndarray, phi: np.ndarray) -> np.ndarray:
        mean_a = self.h * self.h * float((aq * self.w2).sum())
        return (mean_a * (np.eye(2) if self.M is None else self.M) - b.T @ phi) / self.L ** 2

    def quad_points(self):
        n, h = self.n, self.h
        ex, ey = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        X1, X2 = [], []
        for typ in (0, 1):
            off = self.offsets[typ * self.nq:(typ + 1) * self.nq]
            X1.append((h * (ex.reshape(-1, 1) + off[None, :, 0])).ravel())
            X2.append((h * (ey.reshape(-1, 1) + off[None, :, 1])).ravel())
        W = np.broadcast_to(h * h * self.w2, (2 * n * n, self.nq)).ravel()
        return np.concatenate(X1), np.concatenate(X2), W

    def eval_grad(self, phi: np.ndarray, x1: np.ndarray, x2: np.ndarray, chunk: int = 200_000):
        """Gradients of the FE functions phi (Ndof, m) at arbitrary points; returns (P, m, 2)."""
        p, n, h = self.p, self.n, self.h
        out = np.empty((len(x1), phi.shape[1], 2))
        for s in range(0, len(x1), chunk):
            u, v = x1[s:s + chunk] / h, x2[s:s + chunk] / h
            ex, ey = np.floor(u), np.floor(v)
            fx, fy = u - ex, v - ey
            cell = (ex.astype(np.int64) % n) * n + (ey.astype(np.int64) % n)
            _, Dx, Dy = _mono_eval(p, fx, fy)
            typ = (fy > fx).astype(np.int64)
            for t in (0, 1):
                sel = np.nonzero(typ == t)[0]
                if len(sel) == 0:
                    continue
                C, conn = self.types[t]["C"], self.types[t]["conn"]
                coef = phi[conn[cell[sel]]]                                  # (P, nloc, m)
                out[s + sel, :, 0] = np.einsum("Pa,Pam->Pm", Dx[sel] @ C, coef) / h
                out[s + sel, :, 1] = np.einsum("Pa,Pam->Pm", Dy[sel] @ C, coef) / h
        return out


def _stiff_kernel(w, Gx, Gy, M):
    """sum_q w_q grad N_A . M grad N_B (M = None: identity), shape (nq, nloc, nloc)."""
    if M is None:
        return w[:, None, None] * (Gx[:, :, None] * Gx[:, None, :] + Gy[:, :, None] * Gy[:, None, :])
    return w[:, None, None] * (M[0, 0] * Gx[:, :, None] * Gx[:, None, :]
                               + M[0, 1] * (Gx[:, :, None] * Gy[:, None, :] + Gy[:, :, None] * Gx[:, None, :])
                               + M[1, 1] * Gy[:, :, None] * Gy[:, None, :])


def _load_kernels(h, w, Gx, Gy, M):
    """Kernels of b_i = -int s (M e_i) . grad N_A, i = 1, 2 (M = None: identity)."""
    if M is None:
        return [-h * w[:, None] * Gx, -h * w[:, None] * Gy]
    return [-h * w[:, None] * (M[i, 0] * Gx + M[i, 1] * Gy) for i in range(2)]


def make_mesh(L: float, n: int, p: int, element: str = "Q", M=None):
    return PpMesh(L, n, p, M) if element == "P" else QpMesh(L, n, p, M)


# --------------------------------------------------------------------------------------
# Conjugate gradients, several right-hand sides at once (independent CG per column)
# --------------------------------------------------------------------------------------
def cg(A: sp.csr_matrix, B: np.ndarray, rtol: float, maxiter: int):
    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()
    rr = np.einsum("ij,ij->j", R, R)
    tol2 = rtol ** 2 * rr
    active = rr > tol2
    it = 0
    while active.any() and it < maxiter:
        Q = A @ P
        pq = np.einsum("ij,ij->j", P, Q)
        alpha = np.divide(rr, pq, out=np.zeros_like(rr), where=active)
        X += alpha * P
        R -= alpha * Q
        rr_new = np.einsum("ij,ij->j", R, R)
        beta = np.divide(rr_new, rr, out=np.zeros_like(rr), where=active)
        P = R + beta * P
        rr = rr_new
        active = rr > tol2
        it += 1
    if active.any():
        raise RuntimeError(f"CG did not converge in {maxiter} iterations")
    return X, it


# --------------------------------------------------------------------------------------
# Level hierarchy of the alternating scheme (Chapter 3)
# --------------------------------------------------------------------------------------
@dataclass
class Term:
    kind: str          # "base", "dh" or "dT"
    k: int             # level index
    n_fine: int        # mesh of the first (finer / larger-T) evaluation
    T_fine: float
    n_coarse: int      # mesh of the subtracted evaluation (= n_fine for dT; unused for base)
    T_coarse: float
    N: int             # number of samples, Eq. (11)

    @property
    def label(self):
        return "base" if self.kind == "base" else f"{'Δh' if self.kind == 'dh' else 'ΔT'}_{self.k}"


def hierarchy(L: int, cfg: Config) -> list[Term]:
    """Terms of Eq. (5) with h_F = L^{-1/p}, zeta = 2^{1/p}, T_k = 4^k, k_F = 2 log_4 L.

    The mesh at level k has n_k = round(L zeta^k) cells per side (L/h_k must be an integer
    on the torus); sample sizes use the nominal h_k = zeta^{-k} in Eq. (11):
    N_bc = L^2,  N^(h)_k = ceil(L^2 h_k^{2p}),  N^(T)_k = ceil(L^2 h_{k+1}^{2p}).
    """
    K = int(round(math.log2(L)))
    if 2 ** K != L:
        raise ValueError("L must be a power of 2 so that 2 log_4 L is an integer")
    p = cfg.p
    zeta = 2.0 ** (1.0 / p)                  # = 2^{-log_L h_F} with h_F = L^{-1/p}
    hk = lambda k: zeta ** (-k)
    nk = lambda k: int(round(L * zeta ** k))
    ceil = lambda x: int(math.ceil(x - 1e-9))   # h_k^{2p} = 4^{-k} exactly; ignore rounding noise
    Nbc = ceil(L ** 2)
    terms = [Term("base", 0, nk(0), 1.0, nk(0), 1.0, Nbc)]
    for k in range(K):
        terms.append(Term("dh", k, nk(k + 1), 4.0 ** k, nk(k), 4.0 ** k,
                          ceil(L ** 2 * hk(k) ** (2 * p))))
        terms.append(Term("dT", k, nk(k + 1), 4.0 ** (k + 1), nk(k + 1), 4.0 ** k,
                          ceil(L ** 2 * hk(k + 1) ** (2 * p))))
    return terms


# --------------------------------------------------------------------------------------
# One sample of one term (runs inside worker processes)
# --------------------------------------------------------------------------------------
_FIELDS: dict = {}
_MESHES: dict = {}


def _get_field(L, cfg):
    key = (L, cfg)
    if key not in _FIELDS:
        _FIELDS[key] = GaussianCoefficient(L, cfg)
    return _FIELDS[key]


def _get_mesh(L, n, p, element="Q", M=None):
    key = (L, n, p, element, None if M is None else tuple(np.asarray(M).ravel()))
    if key not in _MESHES:
        _MESHES[key] = make_mesh(L, n, p, element, M)
    return _MESHES[key]


def _solve(mesh: QpMesh, aq, Kdata, T, cfg: Config):
    A = mesh.matrix(Kdata, T)
    b = mesh.rhs(aq)
    rtol = cfg.cg_c * (mesh.h / mesh.L) ** 2
    phi, it = cg(A, b, rtol, cfg.cg_maxiter)
    return mesh.homogenized(aq, b, phi), it, it * mesh.nnz * b.shape[1]


def sample_term(L: int, term: Term, seed: tuple, cfg: Config):
    """Returns (value of first evaluation, value of subtracted evaluation or None, stats)."""
    t0 = time.process_time()
    field = _get_field(L, cfg)
    ghat = field.sample(np.random.default_rng(np.random.SeedSequence(seed)))
    mf = _get_mesh(L, term.n_fine, cfg.p, cfg.element, aniso_matrix(cfg))
    aq_f = mf.coef_at_quad(field, ghat)
    Kd_f = mf.stiffness_data(aq_f)
    a_f, it_f, w_f = _solve(mf, aq_f, Kd_f, term.T_fine, cfg)
    a_c, it_c, w_c = None, 0, 0
    if term.kind == "dh":
        mc = _get_mesh(L, term.n_coarse, cfg.p, cfg.element, aniso_matrix(cfg))
        aq_c = mc.coef_at_quad(field, ghat)
        a_c, it_c, w_c = _solve(mc, aq_c, mc.stiffness_data(aq_c), term.T_coarse, cfg)
    elif term.kind == "dT":                     # same mesh and coefficient, smaller T
        a_c, it_c, w_c = _solve(mf, aq_f, Kd_f, term.T_coarse, cfg)
    stats = dict(cpu=time.process_time() - t0, cg_iters=it_f + it_c, cg_work=w_f + w_c)
    return a_f, a_c, stats


def run_batch(args):
    """Worker entry point: several samples of one term of one repetition."""
    L, rep, ti, term, seeds, cfg = args
    out = []
    for s in seeds:
        a_f, a_c, st = sample_term(L, term, s, cfg)
        out.append((a_f, a_c, st, s[-1]))
    return L, rep, ti, out
