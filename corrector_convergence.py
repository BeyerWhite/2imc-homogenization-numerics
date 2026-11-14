"""Cauchy convergence of the full corrector phi^{L,T,h} along the alternating (T, h) path.

For one coefficient realisation the scheme's path is

    (T_0, h_0) -Δh-> (T_0, h_1) -ΔT-> (T_1, h_1) -Δh-> (T_1, h_2) -ΔT-> ... -> (T_K = L^2, h_K = h_F),

with T_k = 4^k, h_k = 2^{-k/p}, K = log2 L (Chapter 3). The path is continued for `extra`
further levels beyond the thesis endpoint; the last state is the reference phi_*. All
correctors are compared in the scaled energy norm

    || grad u ||  :=  ( L^{-2} int_U sum_{i=1,2} |grad u_i|^2 )^{1/2}

on the quadrature points of the reference mesh, together with the coefficient a^{L,T,h}.
Same field, FEM (P_p triangles with --element P, Q_p quadrilaterals with Q) and CG solver (and tolerance) as the estimator in twoimc.py.

    python corrector_convergence.py --element P --p 3 --L 4 8 16 32 --reals 8 8 8 4 --extra 2

writes results/corrector_tri_p{p}.json (P) or results/corrector_p{p}.json (Q).
"""
import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from twoimc import Config, GaussianCoefficient, cg, make_mesh

HERE = os.path.dirname(os.path.abspath(__file__))


def path(L, p, extra):
    """[(kind, k, T, n)] along the alternating path, continued `extra` levels past T = L^2."""
    K = int(round(math.log2(L)))
    zeta = 2.0 ** (1.0 / p)
    nk = lambda k: int(round(L * zeta ** k))
    steps = [("start", 0, 1.0, nk(0))]
    for k in range(K + extra):
        steps.append(("dh", k, 4.0 ** k, nk(k + 1)))
        steps.append(("dT", k, 4.0 ** (k + 1), nk(k + 1)))
    return steps, K


def solve(L, n, T, field, ghat, cfg):
    m = make_mesh(L, n, cfg.p, cfg.element)
    aq = m.coef_at_quad(field, ghat)
    A = m.matrix(m.stiffness_data(aq), T)
    b = m.rhs(aq)
    phi, it = cg(A, b, cfg.cg_c * (m.h / L) ** 2, cfg.cg_maxiter)
    return m, phi, m.homogenized(aq, b, phi), it


def energy_norm(G, W, L):
    return math.sqrt(float(np.einsum("P,Pmc,Pmc->", W, G, G)) / L ** 2)


def one_realisation(args):
    L, r, cfg, extra, seed = args
    t0 = time.process_time()
    steps, K = path(L, cfg.p, extra)
    field = GaussianCoefficient(L, cfg)
    ghat = field.sample(np.random.default_rng(np.random.SeedSequence((seed, L, r))))
    # reference = last state; quadrature of its mesh is used for every comparison
    _, _, Tref, nref = steps[-1]
    mref, phiref, aref, _ = solve(L, nref, Tref, field, ghat, cfg)
    X1, X2, W = mref.quad_points()
    Gref = mref.eval_grad(phiref, X1, X2)
    rows, Gprev = [], None
    for j, (kind, k, T, n) in enumerate(steps):
        if j == len(steps) - 1:
            m, phi, a, it, G = mref, phiref, aref, 0, Gref
        else:
            m, phi, a, it = solve(L, n, T, field, ghat, cfg)
            G = m.eval_grad(phi, X1, X2)
        rows.append(dict(j=j, kind=kind, k=k, T=T, n=n, h=L / n, iters=it, a=a.tolist(),
                         size=energy_norm(G, W, L),
                         dist_ref=energy_norm(G - Gref, W, L),
                         coef_dist_ref=float(np.linalg.norm(a - aref)),
                         cauchy=None if Gprev is None else energy_norm(G - Gprev, W, L)))
        Gprev = G
    for j in range(1, len(rows)):
        rows[j]["coef_cauchy"] = float(np.linalg.norm(np.array(rows[j]["a"]) - np.array(rows[j - 1]["a"])))
    return L, r, K, rows, time.process_time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--reals", type=int, nargs="+", default=[8, 8, 8, 4])
    ap.add_argument("--extra", type=int, default=1, help="levels beyond the thesis endpoint T = L^2, h = h_F")
    ap.add_argument("--cg_c", type=float, default=None, help="override the CG tolerance factor")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    ap.add_argument("--element", choices=["P", "Q"], default="Q")
    args = ap.parse_args()
    cfg = replace(Config(), p=args.p, nu=args.p + 1.5, element=args.element)
    if args.cg_c is not None:
        cfg = replace(cfg, cg_c=args.cg_c)
    out = args.out or os.path.join(HERE, "results", ("corrector_tri_p" if args.element == "P" else "corrector_p") + f"{args.p}.json")
    tasks = [(L, r, cfg, args.extra, args.seed) for L, R in zip(args.L, args.reals) for r in range(R)]
    tasks.sort(key=lambda t: -t[0])                       # largest L first
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(one_realisation, t) for t in tasks]):
            L, r, K, rows, cpu = fut.result()
            res.setdefault(str(L), dict(K=K, reals=[]))["reals"].append(dict(r=r, cpu=cpu, rows=rows))
            print(f"L={L} realisation {r}: {cpu:.0f}s CPU  ({time.time() - t0:.0f}s elapsed)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dict(cfg=cfg.to_dict(), extra=args.extra, results=res), fh)
    print("wrote", out)


if __name__ == "__main__":
    main()
