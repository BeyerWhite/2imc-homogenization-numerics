"""Analyse results/*.npz and draw the error / cost figures.

    python make_plots.py                  # P_p triangle runs: results/tri_p{p}_L*  -> figures/tri_p{p}_*
    python make_plots.py --element Q      # Q_p quadrilateral runs: results/p{p}_L* -> figures/p{p}_*
    python make_plots.py --variants       # robustness variants results/tri_p3_<variant>_L* -> figures/tri_p3_robustness.*

Slopes in L come with 90% bootstrap intervals (runs resampled independently for each L).
"""
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False, "font.size": 10, "font.family": "DejaVu Sans",
        "axes.titlesize": 11, "axes.titleweight": "bold",
    })


# --------------------------------------------------------------------------------------
def load(tag):
    """All box sizes of one configuration; tag is e.g. 'tri_p3', 'p3' or 'tri_p3_beta2'."""
    runs = []
    for js in sorted(glob.glob(os.path.join(RES, f"{tag}_L*.json")), key=lambda f: int(re.findall(r"_L(\d+)", f)[0])):
        meta = json.load(open(js))
        arr = np.load(js[:-5] + ".npz")
        runs.append(analyse(meta, arr))
    return runs


def analyse(meta, arr):
    L, R, terms = meta["L"], meta["reps"], meta["terms"]
    I = np.eye(2)
    aMC = np.zeros((R, 2, 2))
    statvar = 0.0                       # estimated E|a_MC - E a_MC|_F^2 = sum_l V_l / N_l
    # V_l is pooled over all R*N_l samples of level l (runs are i.i.d.), so levels with N_l = 1 count too
    levels, Ds = [], []
    cpu_rep = np.zeros(R)
    work_rep = np.zeros(R)
    for ti, t in enumerate(terms):
        af, ac = arr[f"t{ti}_af"], arr[f"t{ti}_ac"]
        D = af - ac if t["kind"] != "base" else af          # (R, N, 2, 2)
        aMC += D.mean(axis=1)
        Ds.append((D, t["N"]))
        statvar += D.reshape(-1, 2, 2).var(axis=0, ddof=1).sum() / t["N"]
        cpu = arr[f"t{ti}_cpu"]
        cpu_rep += cpu.sum(axis=1)
        work_rep += arr[f"t{ti}_work"].sum(axis=1)
        Dp = D.reshape(-1, 2, 2)
        levels.append(dict(label=t["label"], kind=t["kind"], k=t["k"], N=t["N"], n=t["n_fine"],
                           T=t["T_fine"], h=L / t["n_fine"], h_nom=2 ** (-(t["k"] + (t["kind"] != "dh")) / meta["cfg"]["p"]) if t["kind"] != "base" else 1.0,
                           mean_norm=np.linalg.norm(Dp.mean(axis=0)),
                           mean_se=math.sqrt(Dp.var(axis=0, ddof=1).sum() / len(Dp)),
                           std=math.sqrt(Dp.var(axis=0, ddof=1).sum()),
                           cpu_per_sample=cpu.mean(), cpu_per_rep=cpu.sum() / R,
                           iters=arr[f"t{ti}_iters"].mean()))
    err = np.linalg.norm(aMC - I, axis=(1, 2))           # Frobenius: MSE = stat. var + bias^2
    rng = np.random.default_rng(0)
    boot = [math.sqrt(np.mean(rng.choice(err, R) ** 2)) for _ in range(2000)]
    mean_a = aMC.mean(axis=0)
    return dict(L=L, R=R, p=meta["cfg"]["p"], aMC=aMC, err=err,
                rmse=math.sqrt(np.mean(err ** 2)), rmse_ci=np.percentile(boot, [5, 95]),
                stat_sd=math.sqrt(statvar),
                bias=np.linalg.norm(mean_a - I), bias_se=math.sqrt(statvar / R),
                mean_a=mean_a, cpu=cpu_rep.mean(), cpu_sd=cpu_rep.std(), work=work_rep.mean(),
                cpu_rep=cpu_rep, work_rep=work_rep, Ds=Ds,
                wall=meta["wall_seconds"] / R, levels=levels,
                model=lemma14_model(L, meta["cfg"]["p"], terms))


def lemma14_model(L, p, terms):
    """K_tot of Definition 16 with the per-sample costs of Lemma 15 (constants set to 1)."""
    tot = 0.0
    for t in terms:
        h, T = L / t["n_fine"], t["T_fine"]
        tot += t["N"] * (L / h) ** 2 * math.sqrt(min(L ** 2, T) / h ** 2) * 2 * math.log(max(L, 2))
    return tot


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def boot_slopes(runs, B=1000, seed=0):
    """90% bootstrap intervals of the slopes in L of RMSE, statistical sd, CPU and CG work.
    Each bootstrap replicate resamples the R runs of every L independently, with replacement."""
    rng = np.random.default_rng(seed)
    Ls = [r["L"] for r in runs]
    out = {k: [] for k in ("rmse", "stat_sd", "cpu", "work")}
    for _ in range(B):
        vals = {k: [] for k in out}
        for r in runs:
            idx = rng.integers(r["R"], size=r["R"])
            vals["rmse"].append(math.sqrt(np.mean(r["err"][idx] ** 2)))
            vals["stat_sd"].append(math.sqrt(sum(D[idx].reshape(-1, 2, 2).var(axis=0, ddof=1).sum() / N for D, N in r["Ds"])))
            vals["cpu"].append(r["cpu_rep"][idx].mean())
            vals["work"].append(r["work_rep"][idx].mean())
        for k in out:
            out[k].append(slope(Ls, vals[k]))
    return {k: np.percentile(v, [5, 95]) for k, v in out.items()}


def fmt_ci(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


# --------------------------------------------------------------------------------------
def plot_all(p, runs, tag, el, cis):
    """tag: file prefix (e.g. 'tri_p3'); el: element label for titles (e.g. 'P3 triangles')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    os.makedirs(OUT, exist_ok=True)
    Ls = np.array([r["L"] for r in runs], float)
    LL = np.geomspace(Ls[0] * 0.85, Ls[-1] * 1.15, 50)

    # ---- 1. error vs L --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for r in runs:
        ax.scatter(np.full(r["R"], r["L"]) * np.exp(np.random.default_rng(1).uniform(-.06, .06, r["R"])),
                   r["err"], s=8, color=COLORS[0], alpha=0.18, lw=0, zorder=1)
    rm = np.array([r["rmse"] for r in runs])
    ci = np.array([r["rmse_ci"] for r in runs]).T
    s_rm = slope(Ls, rm)
    ax.errorbar(Ls, rm, yerr=[rm - ci[0], ci[1] - rm], fmt="o-", color=COLORS[0], capsize=3, zorder=3,
                label=f"RMS error  E[|a_MC − ā|²]^½   (slope {s_rm:.2f}, 90% CI {fmt_ci(cis['rmse'])})")
    sd = np.array([r["stat_sd"] for r in runs])
    ax.plot(Ls, sd, "s--", color=COLORS[1], lw=1.5, ms=6, zorder=2,
            label=f"statistical part  (Σ Var/N)^½   (slope {slope(Ls, sd):.2f}, 90% CI {fmt_ci(cis['stat_sd'])})")
    b, bse = np.array([r["bias"] for r in runs]), np.array([r["bias_se"] for r in runs])
    ax.errorbar(Ls, b, yerr=bse, fmt="^:", color=COLORS[2], lw=1.5, ms=6, capsize=3, zorder=2,
                label="bias  |mean a_MC − ā|  (±1 s.e.)")
    ax.plot(LL, rm[1] * (LL / Ls[1]) ** -2, color=MUTED, lw=1, ls="-", zorder=0)
    xl = math.sqrt(Ls[0] * Ls[1])
    ax.annotate("L⁻² (Thm 17)", (xl, rm[1] * (xl / Ls[1]) ** -2), color=INK2, fontsize=9,
                xytext=(6, 8), textcoords="offset points", ha="left")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(Ls); ax.set_xticklabels([str(int(x)) for x in Ls])
    ax.set_xlabel("box size L"); ax.set_ylabel("error in homogenized matrix (Frobenius norm)")
    ax.set_title(f"Accuracy of the 2IMC estimator, {el}, reference ā = I" + ("" if p >= 3 else "\n(p < d+1: outside Thm 17)"))
    ax.legend(fontsize=8, loc="lower left")
    fig.text(0.01, 0.005, "faint dots: individual estimator runs;  "
             + ", ".join(f"L={int(r['L'])}: {r['R']} runs" for r in runs), fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, f"{tag}_error_vs_L.png"), dpi=160)
    plt.close(fig)

    # ---- 2. cost vs L ----------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    cpu = np.array([r["cpu"] for r in runs])
    work = np.array([r["work"] for r in runs])
    model = np.array([r["model"] for r in runs])
    pred_exp = 4 + max(0.0, (3 - p) / p)          # finest level dominates when p < d + 1
    for ax, y, ylabel, title, ck in [
            (axs[0], cpu, "CPU seconds per estimator run", "Measured compute time", "cpu"),
            (axs[1], work, "CG work per run  (Σ nnz × iterations × RHS)", "Machine-independent work", "work")]:
        ax.plot(Ls, y, "o-", color=COLORS[0], label=f"measured (slope {slope(Ls, y):.2f}; last step {slope(Ls[-2:], y[-2:]):.2f})")
        ax.plot(Ls, model * y[-1] / model[-1], "s--", color=COLORS[1], lw=1.5, ms=6,
                label=f"Lemma 15 + Def. 16 cost model (slope {slope(Ls, model):.2f})")
        ref_thm = LL ** 4 * np.log(LL) ** 2
        ax.plot(LL, ref_thm * y[-1] / (Ls[-1] ** 4 * math.log(Ls[-1]) ** 2), color=MUTED, lw=1,
                label="L⁴ log²L  (Thm 17, Eq. 10)" + ("" if p >= 3 else ";  Thm 17 needs p ≥ 3"))
        if p < 3:
            ref_geo = LL ** pred_exp * np.log(LL)
            ax.plot(LL, ref_geo * y[-1] / (Ls[-1] ** pred_exp * math.log(Ls[-1])), color=MUTED, lw=1, ls=":",
                    label=f"$L^{{{pred_exp:.2f}}}\\log L$  (finest level dominates for p < d+1)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ls); ax.set_xticklabels([str(int(x)) for x in Ls])
        ax.set_xlabel("box size L"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Cost of one 2IMC estimate, {el}, unpreconditioned CG", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{tag}_cost_vs_L.png"), dpi=160)
    plt.close(fig)

    # ---- 3. cost breakdown per level -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, r in enumerate(runs):
        labels = [lv["label"] for lv in r["levels"]]
        share = np.array([lv["cpu_per_rep"] for lv in r["levels"]]) / r["cpu"]
        ax.plot(range(len(share)), share, "o-", color=COLORS[i], label=f"L = {r['L']}")
    longest = max(runs, key=lambda r: len(r["levels"]))
    ax.set_xticks(range(len(longest["levels"])))
    ax.set_xticklabels([lv["label"] for lv in longest["levels"]], rotation=45, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("share of total CPU time")
    ax.set_title(f"Where the time goes: cost share per term of Eq. (5), {el}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{tag}_cost_breakdown.png"), dpi=160)
    plt.close(fig)

    # ---- 4. level statistics for the largest L --------------------------------------
    r = runs[-1]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind, name in [(axs[0], "dh", "Δh"), (axs[1], "dT", "ΔT")]:
        lv = [x for x in r["levels"] if x["kind"] == kind]
        k = np.array([x["k"] for x in lv])
        std = np.array([x["std"] for x in lv])
        mn = np.array([x["mean_norm"] for x in lv]); se = np.array([x["mean_se"] for x in lv])
        ax.plot(k, std, "o-", color=COLORS[0], label=f"std of {name} sample (fit: $2^{{{np.polyfit(k, np.log2(std), 1)[0]:.2f}k}}$)")
        ax.errorbar(k, mn, yerr=se, fmt="s--", color=COLORS[1], lw=1.5, ms=6, capsize=3, label=f"|E {name}|  (±1 s.e.)")
        kk = np.array([k[0], k[-1]])
        if kind == "dh":   # Lemma 18(3): std ~ h_k^p,  h_k = 2^{-k/p} -> 2^{-k}
            ax.plot(kk, std[0] * 2.0 ** (-(kk - k[0])), color=MUTED, lw=1, label="$h_k^p = 2^{-k}$  (Lemma 18.3)")
        else:              # Lemma 18(4): std ~ T^{-1/2} + h^p -> 2^{-k}
            ax.plot(kk, std[0] * 2.0 ** (-(kk - k[0])), color=MUTED, lw=1, label="$T_k^{-1/2} + h_{k+1}^p \\sim 2^{-k}$  (Lemma 18.4)")
        ax.set_yscale("log"); ax.set_xticks(k)
        ax.set_xlabel("level k"); ax.set_title(f"{name} terms, L = {r['L']}")
        ax.legend(fontsize=8)
    fig.suptitle(f"Level statistics of the telescoping differences, {el}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{tag}_level_stats.png"), dpi=160)
    plt.close(fig)

    # ---- 5. error vs cost -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(cpu, rm, yerr=[rm - ci[0], ci[1] - rm], fmt="o-", color=COLORS[0], capsize=3,
                label=f"2IMC, {el}  (error ∝ cost$^{{{slope(cpu, rm):.2f}}}$)")
    for x, y, r in zip(cpu, rm, runs):
        ax.annotate(f"L={r['L']}", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    cc = np.geomspace(cpu[0] * 0.7, cpu[-1] * 1.4, 20)
    ax.plot(cc, rm[-1] * (cc / cpu[-1]) ** -0.5, color=MUTED, lw=1, label="cost$^{-1/2}$  (Monte Carlo optimum, [Mou19])")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CPU seconds per estimator run"); ax.set_ylabel("RMS error")
    ax.set_title(f"Accuracy versus cost, {el}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{tag}_error_vs_cost.png"), dpi=160)
    plt.close(fig)


def table(p, runs, el="", cis=None):
    lines = [f"p = {p}  {el}",
             f"{'L':>4} {'runs':>5} {'RMSE':>10} {'90% CI':>23} {'stat sd':>10} {'bias':>10} {'±se':>9} "
             f"{'CPU s/run':>10} {'wall s/run':>10} {'CG work':>10}"]
    for r in runs:
        lines.append(f"{r['L']:>4} {r['R']:>5} {r['rmse']:10.3e} [{r['rmse_ci'][0]:.3e}, {r['rmse_ci'][1]:.3e}] "
                     f"{r['stat_sd']:10.3e} {r['bias']:10.3e} {r['bias_se']:9.2e} {r['cpu']:10.1f} {r['wall']:10.2f} {r['work']:10.3e}")
    Ls = [r["L"] for r in runs]
    if len(runs) > 1:
        lines.append(f"slopes in L: RMSE {slope(Ls, [r['rmse'] for r in runs]):.2f}, CPU {slope(Ls, [r['cpu'] for r in runs]):.2f}, "
                     f"CG work {slope(Ls, [r['work'] for r in runs]):.2f}, Lemma-15 model {slope(Ls, [r['model'] for r in runs]):.2f}")
        if cis is not None:
            lines.append("90% bootstrap intervals of the slopes (cost intervals reflect sampling only, not machine load): " + ", ".join(
                f"{name} {fmt_ci(cis[k])}" for k, name in (("rmse", "RMSE"), ("stat_sd", "stat sd"), ("cpu", "CPU"), ("work", "CG work"))))
            lines.append("CG work / (L^4 log^2 L): " + ", ".join(f"L={r['L']}: {r['work'] / (r['L'] ** 4 * math.log(r['L']) ** 2):.3g}" for r in runs))
    r = runs[-1]
    lines.append(f"mean a_MC at L={r['L']}: {np.array2string(r['mean_a'], precision=5)}")
    lines.append(f"levels at L={r['L']}:")
    for lv in r["levels"]:
        lines.append(f"   {lv['label']:>6} n={lv['n']:>4} T={lv['T']:>7.0f} N={lv['N']:>5}  E={lv['mean_norm']:.2e}±{lv['mean_se']:.1e} "
                     f"sd={lv['std']:.2e}  CG it={lv['iters']:.0f}  CPU/sample={lv['cpu_per_sample']:.3f}s  CPU share={lv['cpu_per_rep'] / r['cpu']:.3f}")
    return "\n".join(lines)


def element_label(el, p):
    return f"P{p} triangles" if el == "P" else f"Q{p} elements"


VARIANT_NAMES = {"": "baseline (β=1, ℓ=0.25, ν=4.5)", "_beta2": "β = 2 (contrast e⁴ ≈ 55)", "_beta3": "β = 3 (contrast e⁶ ≈ 400)",
                 "_ell0.125": "ℓ = 0.125", "_ell0.5": "ℓ = 0.5", "_nu2.5": "ν = 2.5 (< p+1)",
                 "_nu1.5_kmax75": "ν = 1.5 (rough)", "_nu0.5_kmax75": "ν = 0.5 (very rough)"}


def robustness(p=3):
    """Compare the baseline triangle run with the robustness variants (test 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    found = {}
    for var in VARIANT_NAMES:
        runs = [r for r in load(f"tri_p{p}{var}") if r["L"] <= 16] if var == "" else load(f"tri_p{p}{var}")
        if len(runs) >= 2:
            found[var] = runs
    lines = [f"Robustness of the 2IMC estimator, P{p} triangles (slopes in L with 90% bootstrap intervals)",
             f"{'configuration':<30} {'L':>4} {'runs':>5} {'RMSE':>10} {'stat sd':>10} {'bias':>10} {'CG work':>10} {'work/(L^4 log^2 L)':>19}"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    for i, (var, runs) in enumerate(found.items()):
        Ls = np.array([r["L"] for r in runs], float)
        cis = boot_slopes(runs, B=500)
        rm = np.array([r["rmse"] for r in runs]); w = np.array([r["work"] for r in runs])
        lab = VARIANT_NAMES[var]
        for r in runs:
            lines.append(f"{lab:<30} {r['L']:>4} {r['R']:>5} {r['rmse']:10.3e} {r['stat_sd']:10.3e} {r['bias']:10.3e} {r['work']:10.3e} "
                         f"{r['work'] / (r['L'] ** 4 * math.log(r['L']) ** 2):19.3g}")
        lines.append(f"{'':<30} slopes: RMSE {slope(Ls, rm):.2f} {fmt_ci(cis['rmse'])}, stat sd {slope(Ls, [r['stat_sd'] for r in runs]):.2f} "
                     f"{fmt_ci(cis['stat_sd'])}, CG work {slope(Ls, w):.2f} {fmt_ci(cis['work'])}")
        c = COLORS[i % len(COLORS)]
        ls = "-" if var == "" else "--"
        axs[0].plot(Ls, rm, "o" + ls, color=c, label=f"{lab}   [RMSE slope {slope(Ls, rm):.2f}]")
        axs[1].plot(Ls, w / (Ls ** 4 * np.log(Ls) ** 2), "o" + ls, color=c, label=lab)
        r = runs[-1]
        lv = [x for x in r["levels"] if x["kind"] == "dh"]
        axs[2].plot([x["k"] for x in lv], [x["std"] for x in lv], "o" + ls, color=c, label=lab)
    LL = np.array([4, 16.0])
    if "" in found:
        r0 = found[""][0]["rmse"]
        axs[0].plot(LL, r0 * (LL / 4) ** -2, color=MUTED, lw=1, label="L⁻² (Thm 17)")
    for ax in axs:
        ax.set_yscale("log")
    for ax in axs[:2]:
        ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16]); ax.set_xticklabels(["4", "8", "16"]); ax.set_xlabel("box size L")
    axs[0].set_ylabel("RMS error"); axs[0].set_title("Accuracy")
    axs[1].set_ylabel("CG work / (L⁴ log²L)"); axs[1].set_title("Cost relative to Eq. (10)")
    axs[2].set_xlabel("level k"); axs[2].set_ylabel("std of Δh_k sample"); axs[2].set_title("Δh level variance at L = 16")
    axs[2].set_xticks(range(0, 1 + max(int(x.get_xdata().max()) for x in axs[2].get_lines())))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Robustness of the 2IMC estimator to contrast β, correlation length ℓ and smoothness ν, P{p} triangles",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(OUT, f"tri_p{p}_robustness.png"), dpi=160)
    plt.close(fig)
    txt = "\n".join(lines)
    open(os.path.join(OUT, f"tri_p{p}_robustness.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--element", choices=["P", "Q"], default="P")
    ap.add_argument("--variants", action="store_true", help="draw the robustness comparison instead")
    args = ap.parse_args()
    if args.variants:
        robustness()
        raise SystemExit
    pre = "tri_" if args.element == "P" else ""
    ps = sorted({int(m) for f in glob.glob(os.path.join(RES, f"{pre}p*_L*.json"))
                 for m in re.findall(rf"^{pre}p(\d+)_L\d+\.json$", os.path.basename(f))})
    for p in ps:
        tag, el = f"{pre}p{p}", element_label(args.element, p)
        runs = load(tag)
        cis = boot_slopes(runs) if len(runs) >= 2 else None
        txt = table(p, runs, el, cis)
        print(txt)
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT, f"{tag}_summary.txt"), "w").write(txt + "\n")
        if len(runs) >= 2:
            plot_all(p, runs, tag, el, cis)
