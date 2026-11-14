"""Sanity checks for the field generator and the FEM before running the estimator.

    python validate.py [--element P|Q] [--plot]

--element Q (default) checks the Q_p quadrilateral elements and draws figures/fem_convergence.png;
--element P checks the P_p triangle elements and draws figures/fem_convergence_tri.png.
"""
import math
import os
import sys

import numpy as np

from twoimc import Config, GaussianCoefficient, QpMesh, cg, make_mesh

ELEMENT = "P" if "--element" in sys.argv and sys.argv[sys.argv.index("--element") + 1] == "P" else "Q"
mesh_of = lambda L, n, p: make_mesh(L, n, p, ELEMENT)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def quad_points(mesh):
    """Physical x1, x2 of the quadrature points, shape (n_el, nq), matching coef_at_quad."""
    X1, X2, _ = mesh.quad_points()
    nq = len(mesh.w2)
    return X1.reshape(-1, nq), X2.reshape(-1, nq)


def solve_hom(mesh, aq, T):
    A = mesh.matrix(mesh.stiffness_data(aq), T)
    b = mesh.rhs(aq)
    phi, it = cg(A, b, 1e-13, 100_000)
    return mesh.homogenized(aq, b, phi)


def check_field():
    cfg = Config()
    L = 8
    f = GaussianCoefficient(L, cfg)
    rng = np.random.default_rng(1)
    print(f"[field] L={L}: Mg={f.Mg}, lattice variance={f.variance:.6f}")
    # exact evaluation vs direct trigonometric sum at the shifted grid
    ghat = f.sample(rng)
    n, shifts = 5, np.array([0.0, 0.3])
    g = f.g_on_grid(ghat, n, shifts)
    k = 2 * math.pi * f.m / L
    for (ix, qx, iy, qy) in [(0, 0, 0, 0), (3, 1, 2, 0), (4, 1, 4, 1)]:
        x1, x2 = L / n * (ix + shifts[qx]), L / n * (iy + shifts[qy])
        direct = (ghat * np.exp(1j * k[:, None] * x1) * np.exp(1j * k[None, :] * x2)).sum()
        assert abs(direct.imag) < 1e-10 and abs(direct.real - g[ix, qx, iy, qy]) < 1e-10, (direct, g[ix, qx, iy, qy])
    # empirical variance and correlation at distance 1
    vals, corr = [], []
    for _ in range(400):
        gg = f.g_on_grid(f.sample(rng), L, np.array([0.0]))[:, 0, :, 0]
        vals.append(gg.ravel())
        corr.append(np.mean(gg * np.roll(gg, 1, axis=0)))
    v = np.concatenate(vals)
    print(f"[field] empirical Var g = {v.var():.4f},  mean = {v.mean():+.4f},  corr(|x|=1) = {np.mean(corr):.3f}")
    a = np.exp(cfg.beta * np.tanh(v))
    print(f"[field] a range in samples: [{a.min():.3f}, {a.max():.3f}];  E[a] = {a.mean():.4f}, E[1/a] = {np.mean(1 / a):.4f}")


def check_constant():
    for p in (1, 2, 3):
        m = mesh_of(4.0, 5, p)
        aq = np.full(quad_points(m)[0].shape, 2.7)
        ah = solve_hom(m, aq, 1.0)
        assert np.allclose(ah, 2.7 * np.eye(2), atol=1e-12), ah
    print(f"[fem {ELEMENT}] constant coefficient reproduces a_hom = c I exactly (p = 1, 2, 3)")


def check_laminate(make_plot=True):
    """a(x) = exp(0.8 sin(2 pi x1 / L) + 0.3 cos(4 pi x1 / L)):  without the mass term
    a_hom = diag(harmonic mean, arithmetic mean). T = 1e12 makes the massive error negligible."""
    L = 4.0
    fun = lambda x1: np.exp(0.8 * np.sin(2 * np.pi * x1 / L) + 0.3 * np.cos(4 * np.pi * x1 / L))
    xs = (np.arange(200000) + 0.5) * L / 200000
    harm, arit = 1 / np.mean(1 / fun(xs)), np.mean(fun(xs))
    res = {}
    for p in (1, 2, 3):
        errs, hs = [], []
        for n in (2, 3, 4, 6, 8, 12, 16, 24, 32):
            m = mesh_of(L, n, p)
            X1, _ = quad_points(m)
            aq = fun(X1)
            ah = solve_hom(m, aq, 1e12)
            errs.append(abs(ah[0, 0] - harm))
            hs.append(m.h)
            quad_mean = m.h ** 2 * (aq * m.w2).sum() / L ** 2      # phi_2 = 0 exactly
            if ELEMENT == "Q":       # tensor quadrature: phi_2 = 0 and a_12 = 0 exactly
                assert abs(ah[1, 1] - quad_mean) < 1e-10 and abs(ah[0, 1]) < 1e-10, ah
        rate = np.polyfit(np.log(hs[-3:]), np.log(errs[-3:]), 1)[0]
        print(f"[fem] laminate, p={p}: |a11 - harmonic mean| rate ~ h^{rate:.2f} (theory h^{2 * p})")
        res[p] = (hs, errs)
    return res


def check_gaussian_cauchy():
    """Cauchy convergence of a^{L,T,h} for one Gaussian realisation (Lemma 18(5): h^{2p})."""
    cfg = Config()
    L, T = 8, 16.0
    f = GaussianCoefficient(L, cfg)
    ghat = f.sample(np.random.default_rng(7))
    res = {}
    for p in (1, 2, 3):
        ns = [4, 6, 8, 12, 16, 24, 32, 96]
        vals = []
        for n in ns:
            m = mesh_of(L, n, p)
            vals.append(solve_hom(m, m.coef_at_quad(f, ghat), T))
        ref = vals[-1]
        hs = [L / n for n in ns[:-1]]
        errs = [np.linalg.norm(v - ref, 2) for v in vals[:-1]]
        rate = np.polyfit(np.log(hs[-4:-1]), np.log(errs[-4:-1]), 1)[0]
        print(f"[fem] Gaussian realisation L={L}, T={T}, p={p}: |a^h - a^ref| rate ~ h^{rate:.2f} (theory h^{2 * p})")
        res[p] = (hs, errs)
    return res


if __name__ == "__main__":
    check_field()
    check_constant()
    lam = check_laminate()
    gau = check_gaussian_cauchy()
    if "--plot" in sys.argv:
        import matplotlib
        import matplotlib.ticker
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from make_plots import style, COLORS
        style()
        os.makedirs(OUT, exist_ok=True)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, res, title in [(axs[0], lam, "Laminate: |a₁₁ − harmonic mean|"),
                               (axs[1], gau, "Gaussian sample (L=8, T=16): $|a^h - a^{h=1/12}|$")]:
            for i, (p, (hs, errs)) in enumerate(res.items()):
                ax.loglog(hs, errs, "o-", color=COLORS[i], label=f"{ELEMENT}{p}")
                h0, e0 = hs[-2], errs[-2]
                hh = np.array([hs[0], hs[-1]])
                ax.loglog(hh, e0 * (hh / h0) ** (2 * p), "--", color=COLORS[i], lw=1, alpha=0.6)
            ax.set_xlabel("mesh size h")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks([0.125, 0.25, 0.5, 1, 2])
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.set_title(title, fontsize=10)
            ax.legend(title="dashed: $h^{2p}$", fontsize=8, title_fontsize=8)
        name = "fem_convergence.png" if ELEMENT == "Q" else "fem_convergence_tri.png"
        fig.suptitle("FEM convergence, " + ("$Q_p$ on quadrilaterals" if ELEMENT == "Q" else "$P_p$ on triangles"),
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name), dpi=150)
        print("wrote figures/" + name)
