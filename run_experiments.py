"""Run R independent copies of the 2IMC estimator for each box size L.

Example:
    python run_experiments.py --element P --p 3 --L 4 8 16 32 --reps 128 64 32 16

--element P (P_p on triangles) writes results/tri_p{p}_L{L}.*, --element Q (Q_p on
quadrilaterals, the default for backward compatibility) writes results/p{p}_L{L}.*.
Robustness variants (--beta, --ell, --nu, --kmax) add a suffix, e.g. tri_p3_beta2_L16.

Each (L, p) writes results/<tag>.npz with every sample's value and cost, plus a JSON
sidecar with the level hierarchy, configuration and wall-clock time. Progress is
checkpointed to results/p{p}_L{L}.partial.npz every few minutes; rerunning the same command
resumes from it, and an (L, p) whose final .json already exists is skipped.
"""
import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")        # one BLAS thread per worker process

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import numpy as np

from twoimc import Config, hierarchy, run_batch

HERE = os.path.dirname(os.path.abspath(__file__))


CHECKPOINT_SECONDS = 300


def file_tag(cfg, L, variant=""):
    return ("tri_" if cfg.element == "P" else "") + f"p{cfg.p}{variant}_L{L}"


def run_L(L, reps, cfg, pool, workers, seed, outdir, variant=""):
    tag = file_tag(cfg, L, variant)
    if os.path.exists(os.path.join(outdir, tag + ".json")):
        print(f"L={L}: results/{tag}.json exists, skipping", flush=True)
        return
    terms = hierarchy(L, cfg)
    store = {ti: dict(af=np.zeros((reps, t.N, 2, 2)), ac=np.zeros((reps, t.N, 2, 2)),
                      cpu=np.zeros((reps, t.N)), iters=np.zeros((reps, t.N), dtype=np.int64),
                      work=np.zeros((reps, t.N)))
             for ti, t in enumerate(terms)}
    partial = os.path.join(outdir, tag + ".partial.npz")
    done_keys, wall_before = set(), 0.0
    if os.path.exists(partial):
        ck = np.load(partial)
        if int(ck["reps"]) == reps and int(ck["seed"]) == seed:
            for ti, d in store.items():
                for k in d:
                    d[k][...] = ck[f"t{ti}_{k}"]
            done_keys = {tuple(k) for k in ck["done"].tolist()}
            wall_before = float(ck["wall"])
            print(f"L={L}: resuming from checkpoint ({len(done_keys)} tasks done)", flush=True)
    tasks = []
    for ti in reversed(range(len(terms))):            # most expensive levels first
        t = terms[ti]
        b = max(1, min(64, t.N // (2 * workers)))
        for rep in range(reps):
            for j in range(0, t.N, b):
                if (rep, ti, j) in done_keys:
                    continue
                seeds = [(seed, L, rep, ti, i) for i in range(j, min(j + b, t.N))]
                tasks.append((L, rep, ti, t, seeds, cfg))
    print(f"L={L}: {len(terms)} terms, samples/rep={sum(t.N for t in terms)}, reps={reps}, tasks left={len(tasks)}", flush=True)
    t0 = time.time()
    futs = [pool.submit(run_batch, task) for task in tasks]
    done, last, last_ck = 0, t0, t0
    for fut in as_completed(futs):
        _, rep, ti, out = fut.result()
        _store(store, rep, ti, out)
        done_keys.add((rep, ti, out[0][3]))
        done += 1
        now = time.time()
        if now - last > 30:
            last = now
            print(f"  L={L}: {done}/{len(tasks)} tasks, {now - t0:.0f}s elapsed", flush=True)
        if now - last_ck > CHECKPOINT_SECONDS and done < len(tasks):
            last_ck = now
            _checkpoint(partial, store, done_keys, reps, seed, wall_before + now - t0)
    wall = wall_before + time.time() - t0
    arrays = {f"t{ti}_{k}": v for ti, d in store.items() for k, v in d.items()}
    np.savez_compressed(os.path.join(outdir, tag + ".npz"), **arrays)
    meta = dict(L=L, reps=reps, seed=seed, workers=workers, wall_seconds=wall,
                cfg=cfg.to_dict(), terms=[asdict(t) | {"label": t.label} for t in terms])
    with open(os.path.join(outdir, tag + ".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    if os.path.exists(partial):
        os.remove(partial)
    cpu = sum(d["cpu"].sum() for d in store.values())
    print(f"L={L}: done in {wall:.1f}s wall, {cpu:.1f}s CPU ({cpu / reps:.1f}s CPU per estimator)", flush=True)


def _checkpoint(path, store, done_keys, reps, seed, wall):
    arrays = {f"t{ti}_{k}": v for ti, d in store.items() for k, v in d.items()}
    tmp = path + ".tmp.npz"
    np.savez(tmp, done=np.array(sorted(done_keys), dtype=np.int64).reshape(-1, 3),
             reps=reps, seed=seed, wall=wall, **arrays)
    os.replace(tmp, path)


def _store(store, rep, ti, out):
    d = store[ti]
    for a_f, a_c, st, i in out:
        d["af"][rep, i] = a_f
        if a_c is not None:
            d["ac"][rep, i] = a_c
        d["cpu"][rep, i] = st["cpu"]
        d["iters"][rep, i] = st["cg_iters"]
        d["work"][rep, i] = st["cg_work"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--reps", type=int, nargs="+", default=[256, 64, 32, 8])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--element", choices=["P", "Q"], default="Q")
    ap.add_argument("--beta", type=float, default=None, help="contrast (default 1)")
    ap.add_argument("--ell", type=float, default=None, help="Matern length (default 0.25)")
    ap.add_argument("--nu", type=float, default=None, help="Matern smoothness (default p + 1.5)")
    ap.add_argument("--kmax", type=float, default=None, help="absolute Fourier cut-off (rough fields)")
    args = ap.parse_args()
    assert len(args.L) == len(args.reps)
    os.makedirs(args.out, exist_ok=True)
    cfg = replace(Config(), p=args.p, nu=args.p + 1.5, element=args.element)    # Def. 12 needs nu > p + 1
    variant = ""
    for name in ("beta", "ell", "nu", "kmax"):
        v = getattr(args, name)
        if v is not None:
            cfg = replace(cfg, **{name: v})
            variant += f"_{name}{v:g}"
    print("config:", cfg, "variant:", variant or "(baseline)", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for L, R in zip(args.L, args.reps):
            run_L(L, R, cfg, pool, args.workers, args.seed, args.out, variant)


if __name__ == "__main__":
    main()
