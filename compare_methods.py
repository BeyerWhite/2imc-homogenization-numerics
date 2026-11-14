"""Cost of the 2IMC estimator against standard alternatives at equal accuracy (tests 1 and 4).

Inputs:
  results/tri_p{p}_L*.{json,npz}      the 2IMC runs (per-level variance V_l and cost C_l, Eq. (11) sizes)
  results/baselines_tri_p{p}.npz      coupled solves along the h- and T-ladders (baselines.py)

All methods below telescope to the same finest state (T_F = L^2, h_F), so they have the same
bias; at equal statistical error eps they have the same RMS error. For each L we take eps^2 =
the statistical variance sum_l V_l / N_l of the 2IMC estimator with the sample sizes of
Eq. (11), and compute the cost each method needs to reach it:

  2IMC, Eq. (11)            sum_l N_l C_l                            (what was run)
  2IMC, optimal N_l         N_l = ceil(eps^-2 sqrt(V_l / C_l) sum_j sqrt(V_j C_j))   (test 4)
  MLMC in h only (T = T_F)  same optimal formula on the h-ladder differences
  MLMC in T only (h = h_F)  same optimal formula on the T-ladder differences
  plain Monte Carlo         N = ceil(V_F / eps^2) samples at (T_F, h_F)

The baselines get the variance-optimal allocation (generous to them); the 2IMC row uses the
thesis's a priori sizes. Costs are CG work (sum nnz x iterations x RHS) and CPU seconds.
Intervals: 90% bootstrap over the baseline realisations and the 2IMC runs.

    python compare_methods.py [--p 3]   ->  figures/tri_p{p}_methods_*.png, tri_p{p}_methods.txt
    python compare_methods.py --variant _nu0.5_kmax75   (needs the matching robustness and baseline runs)
"""
import argparse
import json
import math
import os

import numpy as np

from make_plots import VARIANT_NAMES, COLORS, INK2, MUTED, OUT, RES, load, slope, style


def mlmc_opt_cost(V, C, eps2):
    """Cost of a (multi-)level estimator with variance-optimal sample sizes, at least 1 per level."""
    V, C = np.asarray(V, float), np.asarray(C, float)
    S = np.sum(np.sqrt(V * C))
    N = np.maximum(1, np.ceil(np.sqrt(V / C) * S / eps2))
    return float(np.sum(N * C)), N


def frob_var(X):
    """Total variance E|X - EX|_F^2 of 2x2 samples X (n, 2, 2)."""
    return float(X.reshape(len(X), -1).var(axis=0, ddof=1).sum())


def ladder_levels(a, cost):
    """Level samples and per-sample costs of single-index MLMC along a ladder a[:, k] (k coarse -> fine)."""
    Ys = [a[:, 0]] + [a[:, k] - a[:, k - 1] for k in range(1, a.shape[1])]
    Cs = [cost[:, 0].mean()] + [(cost[:, k] + cost[:, k - 1]).mean() for k in range(1, a.shape[1])]
    return Ys, Cs


def methods_for_L(run, bl, L, costkey, idx_run=None, idx_bl=None):
    """Cost of every method at the 2IMC's statistical error for box size L."""
    # --- 2IMC: per-level V and C from the runs
    V2, C2, N2 = [], [], []
    for (D, N), lv in zip(run["Ds"], run["levels"]):
        Dk = D if idx_run is None else D[idx_run]
        V2.append(frob_var(Dk.reshape(-1, 2, 2)))
        N2.append(N)
    C2 = np.array([lv["cpu_per_sample"] for lv in run["levels"]]) if costkey == "cpu" else np.array(run["work_per_sample"])
    V2, N2 = np.array(V2), np.array(N2)
    eps2 = float(np.sum(V2 / N2))
    out = {"2IMC, Eq. (11)": float(np.sum(N2 * C2))}
    out["2IMC, optimal N"], Nopt = mlmc_opt_cost(V2, C2, eps2)
    # --- baselines
    ah, aT = bl[f"L{L}_h_a"], bl[f"L{L}_T_a"]
    ch, cT = bl[f"L{L}_h_{costkey}"], bl[f"L{L}_T_{costkey}"]
    if idx_bl is not None:
        ah, aT, ch, cT = ah[idx_bl], aT[idx_bl], ch[idx_bl], cT[idx_bl]
    aT = np.concatenate([aT, ah[:, -1:]], axis=1)               # finest state is shared
    cT = np.concatenate([cT, ch[:, -1:]], axis=1)
    Ys, Cs = ladder_levels(ah, ch)
    out["MLMC in h only"], _ = mlmc_opt_cost([frob_var(Y) for Y in Ys], Cs, eps2)
    Ys, Cs = ladder_levels(aT, cT)
    out["MLMC in T only"], _ = mlmc_opt_cost([frob_var(Y) for Y in Ys], Cs, eps2)
    VF, CF = frob_var(ah[:, -1]), ch[:, -1].mean()
    out["plain Monte Carlo"] = float(math.ceil(VF / eps2) * CF)
    return out, eps2, dict(V=V2, C=C2, N=N2, Nopt=Nopt, VF=VF, CF=CF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--variant", default="", help="robustness suffix, e.g. _nu0.5_kmax75")
    args = ap.parse_args()
    p, v = args.p, args.variant
    vn = f", {VARIANT_NAMES[v]}" if v else ""
    runs = {r["L"]: r for r in load(f"tri_p{p}{v}")}
    blz = np.load(os.path.join(RES, f"baselines_tri_p{p}{v}.npz"))
    bl = {k: blz[k] for k in blz.files}
    blmeta = json.load(open(os.path.join(RES, f"baselines_tri_p{p}{v}.json")))
    Ls = [L for L in sorted(runs) if f"L{L}_h_a" in bl]
    for L in Ls:            # CG work per sample for each 2IMC level (mean over all samples of all runs)
        meta = json.load(open(os.path.join(RES, f"tri_p{p}{v}_L{L}.json")))
        arr = np.load(os.path.join(RES, f"tri_p{p}{v}_L{L}.npz"))
        runs[L]["work_per_sample"] = [arr[f"t{ti}_work"].mean() for ti in range(len(meta["terms"]))]

    rng = np.random.default_rng(0)
    res = {}
    for costkey in ("work", "cpu"):
        res[costkey] = {}
        for L in Ls:
            est, eps2, info = methods_for_L(runs[L], bl, L, costkey)
            R, Rb = runs[L]["R"], bl[f"L{L}_h_a"].shape[0]
            boots = {m: [] for m in est}
            for _ in range(args.boot):
                b, _, _ = methods_for_L(runs[L], bl, L, costkey, rng.integers(R, size=R), rng.integers(Rb, size=Rb))
                for m in b:
                    boots[m].append(b[m] / b["2IMC, Eq. (11)"])
            res[costkey][L] = dict(cost=est, eps=math.sqrt(eps2), info=info,
                                   ratio_ci={m: np.percentile(v, [5, 95]) for m, v in boots.items()})
    methods = list(res["work"][Ls[0]]["cost"])

    # ---------------- text summary ----------------
    lines = [f"Cost to reach the accuracy of the 2IMC estimator (Eq. (11) sizes), P{p} triangles{vn}",
             "All methods share the finest state (T_F = L^2, h_F) and hence the bias; eps = 2IMC statistical sd.",
             f"Baseline realisations per L: " + ", ".join(f"L={L}: {blmeta['L'][str(L)]['R']}" for L in Ls), ""]
    for costkey, unit in (("work", "CG work"), ("cpu", "CPU s")):
        lines.append(f"{unit} per estimate (ratio to 2IMC Eq. (11), 90% bootstrap interval)")
        lines.append(f"{'method':<22}" + "".join(f"{'L=' + str(L):>34}" for L in Ls) + f"{'slope':>8}")
        for m in methods:
            cells = []
            for L in Ls:
                c = res[costkey][L]["cost"][m]
                ratio = c / res[costkey][L]["cost"]["2IMC, Eq. (11)"]
                ci = res[costkey][L]["ratio_ci"][m]
                cells.append(f"{c:10.3e} (x{ratio:6.2f} [{ci[0]:.2f}, {ci[1]:.2f}])")
            sl = slope(Ls, [res[costkey][L]["cost"][m] for L in Ls])
            lines.append(f"{m:<22}" + "".join(f"{x:>34}" for x in cells) + f"{sl:8.2f}")
        lines.append("")
    Lm = Ls[-1]
    info = res["work"][Lm]["info"]
    lines.append(f"Sample sizes at L={Lm}: Eq. (11) vs variance-optimal for the same eps (test 4)")
    for lv, N, No, V, C in zip(runs[Lm]["levels"], info["N"], info["Nopt"], info["V"], info["C"]):
        lines.append(f"   {lv['label']:>6}  N_Eq11={N:>6}  N_opt={int(No):>6}   V={V:.2e}  C={C:.2e}")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"tri_p{p}{v}_methods.txt"), "w").write(txt + "\n")

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    marks = {"2IMC, Eq. (11)": ("o-", COLORS[0]), "2IMC, optimal N": ("o--", COLORS[0]),
             "MLMC in h only": ("s-", COLORS[1]), "MLMC in T only": ("^-", COLORS[2]),
             "plain Monte Carlo": ("D-", COLORS[3])}
    LL = np.array(Ls, float)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for m in methods:
        fmt, c = marks[m]
        cost = np.array([res["work"][L]["cost"][m] for L in Ls])
        rm = np.array([runs[L]["rmse"] for L in Ls])
        axs[0].plot(cost, rm, fmt, color=c, mfc="#fcfcfb" if "optimal" in m else c,
                    label=f"{m}  (error ∝ cost$^{{{slope(cost, rm):.2f}}}$)")
        axs[1].plot(LL, cost, fmt, color=c, mfc="#fcfcfb" if "optimal" in m else c,
                    label=f"{m}  (slope {slope(LL, cost):.2f})")
    for x, y, L in zip([res["work"][L]["cost"]["2IMC, Eq. (11)"] for L in Ls], [runs[L]["rmse"] for L in Ls], Ls):
        axs[0].annotate(f"L={L}", (x, y), xytext=(-6, -12), textcoords="offset points", fontsize=8, color=INK2, ha="right")
    ref = LL ** 4 * np.log(LL) ** 2
    c0 = res["work"][Ls[-1]]["cost"]["2IMC, Eq. (11)"]
    axs[1].plot(LL, ref * c0 / ref[-1], color=MUTED, lw=1, label="L⁴ log²L  (Thm 17, Eq. 10)")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("CG work per estimate  (Σ nnz × iterations × RHS)"); axs[0].set_ylabel("RMS error (same for all methods)")
    axs[0].set_title("Error versus cost at equal accuracy")
    axs[1].set_xscale("log", base=2); axs[1].set_yscale("log")
    axs[1].set_xticks(LL); axs[1].set_xticklabels([str(L) for L in Ls])
    axs[1].set_xlabel("box size L"); axs[1].set_ylabel("CG work to reach the 2IMC accuracy")
    axs[1].set_title("Cost versus L")
    axs[0].legend(fontsize=7.5, loc="upper right"); axs[1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"2IMC against single-index MLMC and plain Monte Carlo, P{p} triangles{vn} "
                 "(baselines with variance-optimal sample sizes)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"tri_p{p}{v}_methods_comparison.png"), dpi=160)
    plt.close(fig)

    # sample sizes: Eq. (11) vs optimal, L = largest
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = [lv["label"] for lv in runs[Lm]["levels"]]
    x = np.arange(len(labels))
    ax.plot(x, info["N"], "o-", color=COLORS[0], label="Eq. (11)")
    ax.plot(x, info["Nopt"], "s--", color=COLORS[1], label="variance-optimal for the same statistical error")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("samples per level")
    r = res["work"][Lm]["cost"]
    ax.set_title(f"Sample sizes per term, L = {Lm}, P{p} triangles{vn}\n"
                 f"(optimal allocation costs ×{r['2IMC, optimal N'] / r['2IMC, Eq. (11)']:.2f} of Eq. (11))")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"tri_p{p}{v}_sample_sizes.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
