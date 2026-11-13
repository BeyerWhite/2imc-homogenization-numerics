"""Figure for corrector_convergence.py:  python plot_corrector.py [--p 2]"""
import argparse
import json
import math
import os

import numpy as np

from make_plots import COLORS, INK2, MUTED, OUT, RES, style


def aggregate(res):
    """Per L: arrays over path steps (RMS over realisations, plus min/max)."""
    agg = {}
    for Ls, d in res.items():
        reals = d["reals"]
        rows0 = reals[0]["rows"]
        get = lambda key: np.array([[row[key] if row.get(key) is not None else np.nan for row in r["rows"]] for r in reals])
        agg[int(Ls)] = dict(
            K=d["K"], R=len(reals),
            kind=[row["kind"] for row in rows0], k=np.array([row["k"] for row in rows0]),
            T=np.array([row["T"] for row in rows0]), h=np.array([row["h"] for row in rows0]),
            **{key: get(key) for key in ("cauchy", "dist_ref", "coef_dist_ref", "coef_cauchy", "size")},
            cpu=np.mean([r["cpu"] for r in reals]))
    return dict(sorted(agg.items()))


def rms(x):
    return np.sqrt(np.nanmean(x ** 2, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--file", default=None, help="results JSON (default results/corrector_p{p}.json)")
    args = ap.parse_args()
    data = json.load(open(args.file or os.path.join(RES, f"corrector_p{args.p}.json")))
    agg = aggregate(data["results"])
    p, extra = args.p, data["extra"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))

    # ---- top: size of each update, by type ------------------------------------------
    for ax, kind, name, ref_lab in [
            (axs[0, 0], "dh", "Δh", r"$h_k^p = 2^{-k}$"),
            (axs[0, 1], "dT", "ΔT", r"$T_k^{-1/2} = 2^{-k}$")]:
        for i, (L, a) in enumerate(agg.items()):
            sel = np.array([kd == kind for kd in a["kind"]])
            k, y = a["k"][sel], rms(a["cauchy"][:, sel])
            lo, hi = np.nanmin(a["cauchy"][:, sel], 0), np.nanmax(a["cauchy"][:, sel], 0)
            inside = k < a["K"]
            ax.fill_between(k, lo, hi, color=COLORS[i], alpha=0.12, lw=0)
            ax.plot(k, y, "-", color=COLORS[i], lw=2)
            ax.plot(k[inside], y[inside], "o", color=COLORS[i], label=f"L = {L}")
            ax.plot(k[~inside], y[~inside], "o", mfc="#fcfcfb", mec=COLORS[i], mew=2)
        kk = np.arange(0, max(len(a["k"]) for a in agg.values()) // 2 + 1)
        y0 = rms(next(iter(agg.values()))["cauchy"][:, [i for i, kd in enumerate(next(iter(agg.values()))["kind"]) if kd == kind]])[0]
        ax.plot(kk, y0 * 2.0 ** -kk.astype(float), color=MUTED, lw=1, label=ref_lab)
        ax.set_yscale("log")
        ax.set_xticks(kk)
        ax.set_xlabel("level k" + ("   (step $T_k$: $h_k \\to h_{k+1}$)" if kind == "dh" else "   (step $h_{k+1}$: $T_k \\to 4T_k$)"))
        ax.set_ylabel(f"‖∇(φ after − φ before)‖  for a {name} step")
        ax.set_title(f"Size of each {name} update of the corrector")
        ax.legend(fontsize=8, loc="lower left")

    # ---- bottom: distance to the most refined state along the path ------------------
    longest = max(agg.values(), key=lambda a: len(a["kind"]))
    J = np.arange(len(longest["kind"]))
    ticklab = [f"T={int(T)}\nh={h:.2g}" if j % 2 == 0 else "" for j, (T, h) in
               enumerate(zip(longest["T"], [2 ** (-(k + (kd != 'start')) / p) for k, kd in zip(longest["k"], longest["kind"])]))]
    for ax, key, ylab, title in [
            (axs[1, 0], "dist_ref", r"$\|\nabla(\varphi_j - \varphi_*)\|$", r"Corrector: distance to the most refined solution $\varphi_*$"),
            (axs[1, 1], "coef_dist_ref", r"$|a_j - a_*|$  (Frobenius)", r"Homogenized coefficient $a^{L,T,h}$: distance to $a_*$")]:
        for i, (L, a) in enumerate(agg.items()):
            j = np.arange(len(a["kind"]) - 1)                 # last point is the reference itself
            y = rms(a[key][:, :-1])
            lo, hi = np.nanmin(a[key][:, :-1], 0), np.nanmax(a[key][:, :-1], 0)
            ax.fill_between(j, lo, hi, color=COLORS[i], alpha=0.12, lw=0)
            ax.plot(j, y, "-o", color=COLORS[i], ms=5, label=f"L = {L}")
            jF = 2 * a["K"]
            ax.plot([jF], [y[jF]], "o", ms=13, mfc="none", mec=INK2, mew=1.5)
        ax.plot([], [], "o", ms=11, mfc="none", mec=INK2, mew=1.5, label="thesis endpoint (T = L², h = h_F)")
        ax.set_yscale("log")
        ax.set_xticks(J)
        ax.set_xticklabels(ticklab, fontsize=7)
        ax.set_xlabel("step j along the alternating path  (labels: state after each ΔT step)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Convergence of the full corrector along the alternating (T, h) path, Q{p} elements, same CG solver",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.945, f"Lines: RMS over realisations; bands: min–max.  Reference $\\varphi_*$: path continued {extra} levels "
             f"past the thesis endpoint (hollow markers: beyond the endpoint).  Norm: "
             r"$\|\nabla u\|^2 = L^{-2}\int_U \sum_i |\nabla u_i|^2$.", ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"corrector_convergence_p{p}.png"), dpi=150)

    lines = [f"Corrector convergence, Q{p}, reference = {extra} levels past T = L^2, h = h_F"]
    for L, a in agg.items():
        jF = 2 * a["K"]
        lines.append(f"L={L:>3} ({a['R']} realisations, {a['cpu']:.0f}s CPU each):  |grad phi| = {rms(a['size'])[-1]:.3f}")
        lines.append(f"      at thesis endpoint: ||grad(phi - phi_*)|| = {rms(a['dist_ref'])[jF]:.2e},  |a - a_*| = {rms(a['coef_dist_ref'])[jF]:.2e}")
        for kind in ("dh", "dT"):
            sel = [i for i, kd in enumerate(a["kind"]) if kd == kind]
            lines.append(f"      {kind} updates: " + "  ".join(f"{v:.2e}" for v in rms(a["cauchy"][:, sel])))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, f"corrector_convergence_p{p}.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
