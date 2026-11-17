"""Builds report/2imc_numerics_report.pdf, a short summary of the numerics with the key figures.

Needs reportlab (not in the project venv):  pip install reportlab
    python report/make_report.py
All numbers are taken from figures/*.txt, figures/anisotropic/*_summary.txt and README.md as of
the L = 32 runs (Nov 2026); figures are included as produced by make_plots.py / compare_methods.py.
"""
import os

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "2imc_numerics_report.pdf")

# Times New Roman covers the Greek letters, ā, ℓ, −, ≈, ½ used below (not ≲, ∼, ∇, ∈: avoided).
FD = "/System/Library/Fonts/Supplemental/"
for name, f in [("TNR", "Times New Roman.ttf"), ("TNR-B", "Times New Roman Bold.ttf"),
                ("TNR-I", "Times New Roman Italic.ttf"), ("TNR-BI", "Times New Roman Bold Italic.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FD + f))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

INK, MUTED, RULE, HEAD = colors.HexColor("#1a1a1a"), colors.HexColor("#555555"), colors.HexColor("#b8b8b8"), colors.HexColor("#eef1f5")
body = ParagraphStyle("body", fontName="TNR", fontSize=10.5, leading=13.6, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=11.2, alignment=0, spaceAfter=0)
cap = ParagraphStyle("cap", parent=body, fontSize=9, leading=11.4, textColor=MUTED, spaceBefore=3, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=body, fontName="TNR-B", fontSize=13, leading=16, spaceBefore=10, spaceAfter=5, alignment=0)
title = ParagraphStyle("title", parent=body, fontName="TNR-B", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle("sub", parent=body, fontSize=10.5, leading=13.5, alignment=TA_CENTER, textColor=MUTED, spaceAfter=2)

W = A4[0] - 4.4 * cm          # text width


def P(t, st=body):
    # tune sub/superscript size and offset per style (reportlab lowers a <sub> by its rise)
    fs = st.fontSize
    t = t.replace("<sub>", f'<sub rise="{0.18 * fs:.2f}" size="{0.72 * fs:.2f}">')
    t = t.replace("<super>", f'<super rise="{0.36 * fs:.2f}" size="{0.72 * fs:.2f}">')
    return Paragraph(t, st)


def bullets(items):
    return ListFlowable([ListItem(P(t), leftIndent=12, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontName="TNR", spaceAfter=4)


def fig(path, frac, caption):
    p = os.path.join(ROOT, path)
    w, h = Image.open(p).size
    width = W * frac
    im = RLImage(p, width=width, height=width * h / w)
    return KeepTogether([im, P(caption, cap)])


def table(rows, widths, caption=None, bold_first_col=False):
    data = [[P(c, small) for c in r] for r in rows]
    t = Table(data, colWidths=[W * x for x in widths], hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    out = [t]
    if caption:
        out.append(P(caption, cap))
    else:
        out.append(Spacer(1, 8))
    return KeepTogether(out)


def L(e):                      # L^e with a proper superscript
    return f"L<super>{e}</super>"


L4 = "L<super>4</super>log<super>2</super>L"
abar = "ā"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("TNR", 8.5); canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"2IMC numerics: summary report  ·  {doc.page}")
    canvas.restoreState()


s = []
s += [P("Numerical validation of the two-index Monte Carlo estimator", title),
      P("Summary of the d = 2 numerics for B. C. White, <i>Two-Index Monte Carlo Method for Approximating "
        "Coefficients of Linear Elliptic Problems</i> (ISTA MSc thesis, 2026)", sub),
      P("November 2026 · code, data and full tables: repository <font face='TNR-I'>2imc-homogenization-numerics</font> (README.md)", sub),
      Spacer(1, 10)]

s += [P("Main findings", h1), bullets([
    f"<b>Theorem 17 is confirmed for P3 elements on triangles.</b> The RMS error decays like {L('−1.99')} "
    f"(90% interval [−2.10, −1.90]) and the CG work like {L('4.80')}, with work/({L4}) flat between 6.2 and 7.4 × 10<super>4</super> for L = 4–32.",
    "<b>The 2IMC beats plain Monte Carlo and single-index MLMC in T</b> by a margin that grows with L "
    "(7.5× and 3.7× less work at L = 32). Against single-index MLMC in h it ties for the smooth field "
    "and wins for rough fields (0.71× and 0.58× of its work at L = 32 for ν = 1.5 and 0.5).",
    "<b>The a priori sample sizes of Eq. (11) are about 3–4× more expensive than variance-optimal ones,</b> "
    "because the bound on the Δh variances is pessimistic by orders of magnitude. The scaling in L is unchanged.",
    "<b>Cost is robust</b> to contrast, correlation length and smoothness: work/(" + L4 + ") stays flat in all "
    "seven variants. The error rate is robust to all but roughness: for ν = 1.5 the discretisation bias "
    f"decays only like {L('−1.15')}, and for ν = 0.5 the mesh does not resolve the field.",
    f"<b>A full anisotropic tensor {abar} = M is recovered</b> with the same rates and cost, in every entry including the off-diagonal one.",
])]

s += [P("1  Set-up", h1), P(
    "The coefficient is a(x) = exp(β tanh g(x)) I on the torus [0, L)<super>2</super>, with g a stationary Gaussian "
    "field with Matérn-type spectrum S(k) = c (1 + ℓ<super>2</super>|k|<super>2</super>)<super>−(ν+1)</super>; by default β = 1, "
    "ℓ = 0.25 and ν = p + 1.5. By symmetry and Keller–Dykhne duality the exact homogenized coefficient is "
    f"{abar} = I, so errors are measured against an exact value. The corrector problem with massive term 1/T is "
    "discretised with periodic P<sub>p</sub> Lagrange elements on a uniform triangulation and solved by unpreconditioned CG. "
    "The estimator follows Definition 13 and Eq. (11): T<sub>k</sub> = 4<super>k</super> up to T<sub>F</sub> = L<super>2</super>, "
    "h<sub>F</sub> = L<super>−1/p</super>, alternating Δh and ΔT levels, sample sizes L<super>2</super>/4<super>k</super>. "
    "For each L = 4, 8, 16, 32 the whole estimator is run 128, 64, 32, 16 times independently. Cost is measured "
    "machine-independently as CG work (Σ nonzeros × iterations × right-hand sides); 90% intervals are bootstrap intervals."),
    P("Degree p = 3 is the main case, since Theorem 17 needs p ≥ d + 1. P2 triangles and Q2/Q3 quadrilaterals were "
      "run as well and are reported in the README; Q3 gives the same errors as P3 at about 1.5× the work.")]

s += [P("2  Accuracy and cost (Theorem 17)", h1),
      table([["", "P3 (main case)", "P2 (p < d + 1)", "Theorem 17"],
             ["RMS error, slope in L", "−1.99 [−2.10, −1.90]", "−2.01 [−2.18, −1.90]", "−2 (up to logs), Eq. (9)"],
             ["statistical part, slope", "−1.94 [−1.96, −1.93]", "−1.96 [−1.97, −1.95]", ""],
             ["RMS error at L = 4, 8, 16, 32", "1.3e-1, 3.4e-2, 8.3e-3, 2.2e-3", "", ""],
             ["CG work, slope in L", "4.80", "4.98", f"{L4}, Eq. (10)"],
             [f"CG work / ({L4}), L = 4…32", "7.1, 7.4, 6.6, 6.2 (×10<super>4</super>)", "2.0, 2.1, 2.3, 2.5 (×10<super>4</super>)", "bounded"]],
            [0.27, 0.27, 0.24, 0.22],
            "Table 1. Error and cost of the 2IMC estimator against L. Brackets: 90% bootstrap intervals."),
      fig("figures/tri_p3_error_vs_L.png", 0.62,
          "Figure 1. RMS error of the P3 estimator against L, split into its statistical part and the bias; "
          "faint dots are the individual runs. The grey line is L<super>−2</super>."),
      fig("figures/tri_p3_cost_vs_L.png", 0.95,
          f"Figure 2. Cost of one estimate: CPU time (left) and CG work (right), with the Lemma 15 / Definition 16 cost "
          f"model and {L4}. The work tracks {L4} closely; the CPU slope is lower because small solves are overhead-dominated."),
      P(f"The statistical part dominates the error; the bias is about half of it, and at L = 32 the averaged estimate is "
        f"I + 8·10<super>−4</super>. No single level dominates the cost (at L = 32 the shares range from 2% to 20%), which is the "
        f"balance the proof of Theorem 17 relies on. For P2, which lies outside the theorem, work/({L4}) grows by 23% over "
        f"L = 4–32, consistent with the L<super>4.5</super> log L cost expected for p < d + 1.")]

s += [P("3  Comparison with standard methods", h1), P(
    "For 256, 128, 64, 32 coefficient realisations the corrector was solved along two one-parameter ladders "
    "that end at the 2IMC's finest state (T<sub>F</sub>, h<sub>F</sub>): the mesh ladder at fixed T = L<super>2</super> "
    "(single-index MLMC in h) and the regularisation ladder on the finest mesh (single-index MLMC in T); the finest "
    "state alone is plain Monte Carlo. All methods therefore share the bias, and each is given the cost needed to reach "
    "the 2IMC's statistical error. The baselines get variance-optimal sample sizes, which favours them."),
    table([["method", "L = 4", "L = 8", "L = 16", "L = 32", "slope"],
           ["2IMC, Eq. (11) sizes", "1", "1", "1", "1", "4.80"],
           ["2IMC, optimal sizes", "0.36", "0.30", "0.26", "0.25", "4.63"],
           ["MLMC in h only", "0.27", "0.23", "0.23", "0.28 [0.19, 0.34]", "4.81"],
           ["MLMC in T only", "0.65", "1.20", "2.08", "3.71 [2.65, 4.43]", "5.64"],
           ["plain Monte Carlo", "0.62", "1.33", "2.70", "7.48 [4.85, 9.38]", "5.98"]],
          [0.26, 0.11, 0.11, 0.11, 0.25, 0.16],
          "Table 2. CG work to reach the 2IMC's accuracy, relative to the 2IMC with Eq. (11) sizes (smooth field, ν = 4.5)."),
    fig("figures/tri_p3_methods_comparison.png", 0.98,
        "Figure 3. Error against cost at equal accuracy (left) and cost against L (right) for the smooth field."),
    P("Plain MC and MLMC in T solve every sample on the finest mesh and scale like L<super>5.6–6.0</super>. MLMC in h "
      "is on par with the 2IMC here because, for this smooth field and P3, the mesh corrections carry almost no variance "
      "(Var Δh<sub>0</sub> = 2.7e-8 against 3.0e-3 for the base level), so its work is essentially one plain MC at "
      "T = L<super>2</super> on the coarsest mesh. The 2IMC's base level at T = 1 is 5.5× cheaper per sample, but the ΔT levels "
      "that remove the regularisation cost about as much again."),
    P("<b>Sample sizes.</b> With the measured level variances, the variance-optimal allocation reaches the same "
      "statistical error at 0.25× (L = 32) to 0.36× (L = 4) of the cost of Eq. (11). Eq. (11) puts far too many samples on "
      "the Δh levels (e.g. 1024 instead of 3 on Δh<sub>0</sub> at L = 32): the bound std(Δh<sub>k</sub>) ≤ C h<sub>k</sub><super>p</super> "
      "of Lemma 18.3 has the right rate (measured 2<super>−1.39k</super> versus 2<super>−k</super>) but a very pessimistic constant. "
      "The ΔT sizes are within a factor 2–4 of optimal. A pilot run, as in standard MLMC, would recover this factor.")]

s += [P("4  Robustness in contrast, correlation length and smoothness", h1), P(
    "Seven variants of the field, each run with the same L and run counts as the main study. "
    f"{abar} = I holds exactly for all of them. For ν ≤ 1.5 the Fourier series is cut at |k| ≤ 75."),
    table([["configuration", "RMSE slope [90%]", "stat. sd slope", "RMSE, L = 32", f"work/({L4}), L = 4…32 (×10<super>4</super>)"],
           ["baseline (β = 1, ℓ = 0.25, ν = 4.5)", "−1.99 [−2.10, −1.90]", "−1.94", "2.2e-3", "7.1, 7.4, 6.6, 6.2"],
           ["β = 2 (contrast ≈ 55)", "−1.86 [−1.96, −1.78]", "−1.86", "9.3e-3", "10.8, 12.1, 10.9, 10.4"],
           ["β = 3 (contrast ≈ 400)", "−1.83 [−1.91, −1.76]", "−1.81", "2.6e-2", "16.2, 19.8, 18.0, 17.2"],
           ["ℓ = 0.125", "−1.84 [−1.97, −1.74]", "−1.99", "1.3e-3", "6.9, 6.9, 6.1, 5.8"],
           ["ℓ = 0.5", "−1.88 [−1.99, −1.79]", "−1.86", "6.5e-3", "5.4, 6.9, 6.8, 6.7"],
           ["ν = 2.5 (violates ν > p + 1)", "−1.94 [−2.12, −1.81]", "−1.96", "1.6e-3", "7.1, 7.3, 6.4, 6.1"],
           ["ν = 1.5 (rough)", "−1.54 [−1.63, −1.46]", "−1.97", "3.3e-3", "6.9, 7.0, 6.2, 5.9"],
           ["ν = 0.5 (very rough)", "−0.46 [−0.49, −0.43]", "−1.85", "3.5e-2", "6.2, 6.2, 5.6, 5.4"]],
          [0.29, 0.2, 0.12, 0.12, 0.27],
          "Table 3. Robustness of the 2IMC estimator (P3 triangles, L = 4–32)."),
    fig("figures/tri_p3_robustness.png", 1.0,
        "Figure 4. Robustness study: RMS error (left), work relative to the Theorem 17 bound (middle) and the "
        "standard deviation of the Δh levels at L = 32 (right)."),
    bullets([
        "<b>Cost is robust.</b> Work/(" + L4 + ") is flat or slightly falling in every variant. Contrast raises the "
        "constant (1.7× for β = 2, 2.8× for β = 3, from worse conditioning) but not the rate.",
        "<b>Contrast and correlation length</b> change the constants and lower the fitted slopes to −1.83…−1.88. "
        "For β this sits in the statistical part: the ΔT levels take a growing share of the variance (β = 3: 52% at L = 16, "
        "67% at L = 32). Under Eq. (11) each ΔT level contributes about equally, so their total grows like log L, "
        "which is the (log L)<super>C</super> allowed in Eq. (9) rather than a different power.",
        "<b>Smoothness.</b> The condition ν > p + 1 is sufficient but not sharp: ν = 2.5 behaves like the baseline. At "
        f"ν = 1.5 the statistical part still decays like {L('−2')}, but the bias only like {L('−1.15')} and dominates at "
        "L = 32. At ν = 0.5 the bias stagnates (slope −0.4) because the field varies on scales far below the finest "
        "mesh (h<sub>F</sub> ≈ 0.31 at L = 32). The loss is entirely in the discretisation bias, the part the smoothness "
        "assumption controls.",
    ])]

s += [P("5  Comparison with standard methods for rough fields", h1), P(
    "Section 3 repeated for ν = 1.5 and ν = 0.5. Once the mesh corrections carry real variance, MLMC in h has to pay for "
    "them at T = L<super>2</super>, where every solve needs many CG iterations, while the 2IMC takes its Δh steps at small T, "
    "where solves are cheap."),
    table([["method", "ν = 4.5", "ν = 1.5", "ν = 0.5", "slope, ν = 1.5 / 0.5"],
           ["2IMC, Eq. (11) sizes", "1", "1", "1", "4.79 / 4.80"],
           ["2IMC, optimal sizes", "0.25", "0.27 [0.27, 0.28]", "0.58 [0.57, 0.60]", "4.66 / 4.82"],
           ["MLMC in h only", "0.28", "0.38 [0.27, 0.47]", "1.00 [0.80, 1.15]", "4.93 / 5.15"],
           ["MLMC in T only", "3.71", "2.69 [1.83, 3.39]", "1.10 [0.73, 1.43]", "5.47 / 5.10"],
           ["plain Monte Carlo", "7.48", "6.66 [4.12, 9.07]", "3.54 [2.15, 4.82]", "5.94 / 5.71"]],
          [0.25, 0.11, 0.19, 0.19, 0.26],
          "Table 4. CG work to reach the 2IMC's accuracy at L = 32, relative to the 2IMC with Eq. (11) sizes."),
    fig("figures/tri_p3_nu1.5_kmax75_methods_comparison.png", 0.98,
        "Figure 5. As Figure 3, for the rough field ν = 1.5."),
    P("With optimal sizes the 2IMC needs 0.71× (ν = 1.5) and 0.58× (ν = 0.5) of the work of MLMC in h at L = 32 and its "
      "cost grows more slowly in L. At ν = 0.5 the lead is clear; at ν = 1.5 the lower end of the MLMC-in-h interval "
      "touches the 2IMC value, so the lead is likely but not certain at the 90% level, although the ratio has fallen "
      "steadily with L (1.28, 0.94, 0.77, 0.71). CPU time gives the same ordering. At ν ≤ 1.5 the RMS error itself is "
      "dominated by the bias shared by all methods; the comparison is at equal statistical error.")]

s += [P("6  Anisotropic coefficient with exact " + abar + " = M", h1), P(
    "To test a full tensor with an off-diagonal entry, the coefficient is a(x) = s(M<super>−1/2</super>x) M, with s the "
    "isotropic scalar coefficient and M = R(30°) diag(2, ½) R(30°)<super>T</super> = [[1.625, 0.650], [0.650, 0.875]], "
    "det M = 1. The substitution x = M<super>1/2</super>y maps the corrector problem to the isotropic one, so "
    f"{abar} = M exactly. The mesh is aligned with neither eigenvector of M. The FEM and the field were validated "
    "separately (constant and laminate coefficients converge to their exact full-tensor values at the h<super>2p</super> rate)."),
    table([["", f"anisotropic, {abar} = M", f"isotropic, {abar} = I"],
           ["RMS error, slope in L", "−2.01 [−2.19, −1.89]", "−1.99 [−2.10, −1.90]"],
           ["RMS error per entry, slope (11, 12, 22)", "−2.02, −1.99, −2.02", ""],
           [f"RMS error at L = 32, relative to |{abar}|", "1.3e-3", "1.5e-3"],
           [f"CG work / ({L4}), L = 4…32 (×10<super>4</super>)", "6.9, 6.9, 6.1, 5.9", "7.1, 7.4, 6.6, 6.2"],
           ["mean estimate at L = 32", "[[1.62615, 0.64976], [0.64976, 0.87547]]", "I + 8e-4"]],
          [0.36, 0.38, 0.26],
          "Table 5. Anisotropic test (P3 triangles, L = 4–32 with 128, 64, 32, 16 runs) against the isotropic baseline."),
    fig("figures/anisotropic/tri_p3_aniso4_theta30_entries.png", 0.98,
        "Figure 6. Entry-wise RMS error (left) and bias (right) of the anisotropic estimate against the exact M."),
    P("Error and cost behave as in the isotropic case, and every entry converges like L<super>−2</super>. The bias "
      "is positive definite at every L and roughly proportional to M, with about the isotropic size; a positive bias is "
      "consistent with the conforming FEM and the massive term, which can only raise the corrector energy. At L = 32 "
      "only the bias of the 11 entry is resolved above the noise.")]

s += [P("7  Caveats and open points", h1), bullets([
    "All results are for d = 2 and L ≤ 32 (three doublings of L, a factor of about 60 in the error). Slopes are fitted over this range; the log factors "
    "of Theorem 17 cannot be separated from the powers at this range, except indirectly (Section 4).",
    "The cost is measured as CG work with unpreconditioned CG, as Lemma 15 assumes; CPU times give the same "
    "conclusions but depend on machine load.",
    "The method comparison uses variance-optimal sizes for the baselines but the a priori Eq. (11) sizes for the "
    "headline 2IMC numbers; with optimal sizes on both sides the 2IMC is clearly ahead of MLMC in h only for rough fields.",
    "For rough fields (ν ≤ 1.5) the accuracy is limited by the discretisation bias at the finest state, which no "
    "variance reduction can remove; a finer h<sub>F</sub> would be needed there.",
])]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                        title="Numerical validation of the two-index Monte Carlo estimator",
                        author="B. C. White", subject="2IMC numerics summary")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
