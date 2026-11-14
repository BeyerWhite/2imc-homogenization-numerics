"""Level statistics of the standard alternatives to the two-index estimator (test 1).

For each L and each of R independent coefficient realisations, solve the corrector problem
along two one-parameter ladders that end at the same finest state (T_F = L^2, h_F = L^{-1/p})
as the 2IMC estimator:

  * h-ladder:  (T_F, h_k),  k = 0..K   (fixed regularisation, refine the mesh)
  * T-ladder:  (T_k, h_F),  k = 0..K   (fixed finest mesh, increase T)

with T_k = 4^k, h_k = L / round(L 2^{k/p}), K = log2 L, exactly the T and h values the
2IMC estimator uses. The finest state is shared by both ladders. From these coupled solves
one gets the per-sample variance and cost of

  * plain Monte Carlo           (every sample at (T_F, h_F)),
  * single-index MLMC in h       (levels along the h-ladder),
  * single-index MLMC in T       (levels along the T-ladder),

which compare_methods.py turns into the cost each method needs to reach the accuracy of the
2IMC estimator. Writes results/baselines_tri_p{p}.npz (or baselines_p{p}.npz for Q_p).

    python baselines.py --element P --p 3 --L 4 8 16 32 --reals 256 128 64 32
    python baselines.py --element P --p 3 --L 4 8 16 --reals 256 128 64 --nu 0.5 --kmax 75
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

from twoimc import Config, _get_field, _get_mesh, _solve, aniso_matrix

HERE = os.path.dirname(os.path.abspath(__file__))


def ladders(L, p):
    K = int(round(math.log2(L)))
    nk = lambda k: int(round(L * 2.0 ** (k / p)))
    TF, nF = 4.0 ** K, nk(K)
    h_ladder = [(TF, nk(k)) for k in range(K + 1)]
    T_ladder = [(4.0 ** k, nF) for k in range(K + 1)]      # T_ladder[K] == h_ladder[K]
    return K, h_ladder, T_ladder


def one_solve(args):
    L, r, which, k, T, n, cfg, seed = args
    t0 = time.process_time()
    field = _get_field(L, cfg)
    ghat = field.sample(np.random.default_rng(np.random.SeedSequence((seed, L, r))))
    m = _get_mesh(L, n, cfg.p, cfg.element, aniso_matrix(cfg))
    aq = m.coef_at_quad(field, ghat)
    a, it, work = _solve(m, aq, m.stiffness_data(aq), T, cfg)
    return L, r, which, k, a, it, work, time.process_time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--element", choices=["P", "Q"], default="P")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--reals", type=int, nargs="+", default=[256, 128, 64, 32])
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None, help="output path without extension")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--ell", type=float, default=None)
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--kmax", type=float, default=None)
    ap.add_argument("--aniso", type=float, default=None)
    ap.add_argument("--theta", type=float, default=None)
    args = ap.parse_args()
    cfg = replace(Config(), p=args.p, nu=args.p + 1.5, element=args.element)
    variant = ""
    for name in ("beta", "ell", "nu", "kmax", "aniso", "theta"):   # same suffixes as run_experiments.py
        val = getattr(args, name)
        if val is not None:
            cfg = replace(cfg, **{name: val})
            variant += f"_{name}{val:g}"
    tag = ("baselines_tri_p" if args.element == "P" else "baselines_p") + str(args.p) + variant
    out = args.out or os.path.join(HERE, "results", tag)
    print("config:", cfg, flush=True)
    store, meta = {}, dict(cfg=cfg.to_dict(), seed=args.seed, L={})
    tasks = []
    for L, R in zip(args.L, args.reals):
        K, hl, Tl = ladders(L, args.p)
        meta["L"][str(L)] = dict(R=R, K=K, h_ladder=hl, T_ladder=Tl)
        for w, lad in (("h", hl), ("T", Tl[:-1])):          # finest state solved once (in the h-ladder)
            store[(L, w)] = dict(a=np.zeros((R, len(lad), 2, 2)), iters=np.zeros((R, len(lad)), np.int64),
                                 work=np.zeros((R, len(lad))), cpu=np.zeros((R, len(lad))))
            for r in range(R):
                for k, (T, n) in enumerate(lad):
                    tasks.append((L, r, w, k, T, n, cfg, args.seed))
    tasks.sort(key=lambda t: -(t[5] ** 2) * math.sqrt(min(t[4], t[0] ** 2)))   # biggest solves first
    t0, last = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one_solve, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            L, r, w, k, a, it, work, cpu = fut.result()
            d = store[(L, w)]
            d["a"][r, k], d["iters"][r, k], d["work"][r, k], d["cpu"][r, k] = a, it, work, cpu
            if time.time() - last > 60:
                last = time.time()
                print(f"  {i + 1}/{len(tasks)} solves, {last - t0:.0f}s elapsed", flush=True)
    arrays = {f"L{L}_{w}_{key}": v for (L, w), d in store.items() for key, v in d.items()}
    np.savez_compressed(out + ".npz", **arrays)
    meta["wall_seconds"] = time.time() - t0
    json.dump(meta, open(out + ".json", "w"), indent=1)
    cpu = sum(d["cpu"].sum() for d in store.values())
    print(f"wrote {out}.npz  ({time.time() - t0:.0f}s wall, {cpu:.0f}s CPU)", flush=True)


if __name__ == "__main__":
    main()
