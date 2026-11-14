# Numerics for the two-index Monte Carlo (2IMC) estimator, d = 2

Checks Theorem 17 of the thesis (B. C. White, *Two-Index Monte Carlo Method for
Approximating Coefficients of Linear Elliptic Problems*, ISTA MSc thesis 2026, final version):
the error bound E|a_MC − ā| ≲ L⁻² (log L)^C (Eq. 9) and the cost bound K_tot ≲ L⁴ log²L
(Eq. 10). In d = 2 the theorem needs p ≥ d + 1 = 3, so degree 3 is the main case and degree 2
is run for comparison. Theorem, lemma and definition numbers below follow the final version.

Two finite-element families are implemented, and every result exists for both:

- **P_p on triangles** (the simplicial elements the thesis analyses): main results, files
  prefixed `tri_` (e.g. `figures/tri_p3_error_vs_L.png`). The comparison with standard
  methods, the variance-optimal sample sizes and the robustness study are run with P3.
- **Q_p on quadrilaterals** (the first implementation): kept unchanged, files without the prefix
  (e.g. `figures/p3_error_vs_L.png`).

## Files

| file | contents |
|---|---|
| `twoimc.py` | the Gaussian field, the periodic P_p (triangles) and Q_p (quadrilaterals) FEM, CG, the level hierarchy and one-sample routines |
| `validate.py` | checks of the field and the FEM (`--aniso`: the anisotropic coefficient); `--plot` draws `figures/fem_convergence.png` (Q_p) or, with `--element P`, `figures/fem_convergence_tri.png` |
| `run_experiments.py` | runs R independent copies of the estimator for each L, in parallel; writes `results/tri_p{p}_L{L}.{npz,json}` (`--element P`) or `results/p{p}_L{L}.*` (`--element Q`, the default). `--beta`, `--ell`, `--nu`, `--kmax` change the coefficient field and add a suffix such as `tri_p3_beta2_L16`; `--aniso`, `--theta` select the anisotropic test (use with `--out results/anisotropic`). It checkpoints every 5 minutes, so rerunning the same command after an interruption resumes, and finished L values are skipped |
| `corrector_convergence.py`, `plot_corrector.py` | Cauchy convergence of the corrector along the (T, h) path for fixed L; writes `results/corrector_{tri_}p{p}.json` and `figures/corrector_convergence_{tri_}p{p}.{png,txt}` |
| `make_plots.py` | analysis with bootstrap intervals, the figures in `figures/` and `figures/{tri_}p{p}_summary.txt`; `--element P` (default) or `Q`; `--variants` draws the robustness figure `figures/tri_p3_robustness.png`; `--aniso` analyses the anisotropic test into `figures/anisotropic/` |
| `baselines.py` | coupled solves along the pure h-ladder and the pure T-ladder that end at the 2IMC's finest state; writes `results/baselines_tri_p{p}.{npz,json}` |
| `compare_methods.py` | cost of plain MC, single-index MLMC in h, single-index MLMC in T and the 2IMC (Eq. 11 and variance-optimal sample sizes) at equal accuracy; writes `figures/tri_p{p}_methods*` and `figures/tri_p{p}_sample_sizes.png`; `--variant _nu0.5_kmax75` repeats it for a robustness field (`baselines.py` takes the same `--nu`/`--kmax`/`--beta`/`--ell` flags) |

```
python -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
.venv/bin/python validate.py --element P --plot
.venv/bin/python run_experiments.py --element P --p 3 --L 4 8 16 32 --reps 128 64 32 16
.venv/bin/python run_experiments.py --element P --p 2 --L 4 8 16 32 --reps 128 64 32 16
.venv/bin/python make_plots.py --element P
.venv/bin/python corrector_convergence.py --element P --p 3 --L 4 8 16 32 --reals 8 8 8 4 --extra 2
.venv/bin/python plot_corrector.py --element P --p 3
.venv/bin/python baselines.py --element P --p 3 --L 4 8 16 32 --reals 256 128 64 32
.venv/bin/python compare_methods.py --p 3
# robustness: one run per variant, then the summary figure
.venv/bin/python run_experiments.py --element P --p 3 --L 4 8 16 32 --reps 128 64 32 16 --beta 2
#   … likewise --beta 3, --ell 0.125, --ell 0.5, --nu 2.5, --nu 1.5 --kmax 75, --nu 0.5 --kmax 75
.venv/bin/python make_plots.py --variants
.venv/bin/python baselines.py --element P --p 3 --L 4 8 16 32 --reals 256 128 64 32 --nu 0.5 --kmax 75   # likewise --nu 1.5
.venv/bin/python compare_methods.py --p 3 --variant _nu0.5_kmax75
```

The Q_p results were made with the same commands and `--element Q` (`plot_corrector.py --element Q`).

## What is implemented

**Coefficient field** (Definition 12, Assumption 4). g is a centred stationary Gaussian field on the
torus [0, L)², built from a Fourier series with Matérn-type spectral density
S(k) ∝ (1 + ℓ²|k|²)^−(ν+1), with ν = p + 1.5 (Definition 12 needs ν > p + 1), ℓ = 0.25
(correlation ≈ 0.29 at |x| = 1) and Var g = 1. Sampling S on the dual lattice 2πℤ²/L is the
same as periodising the whole-space covariance, so the ensemble does not depend on L. This is
the periodisation a_L that Assumption 4 leaves unspecified.
Modes carrying the last 10⁻⁶ of the variance are dropped. (For the rough fields of the
robustness study, ν ≤ 1.5, this would need millions of modes, so `--kmax 75` also cuts at
|k| ≤ 75; this keeps 99.98% of the variance for ν = 1.5 and 94.7% for ν = 0.5.) The coefficient is
a(x) = exp(β tanh g(x)) · I with β = 1. It is C^∞, symmetric and uniformly elliptic
(e⁻¹ ≤ a ≤ e).

The field is evaluated exactly at the quadrature points. For a fixed sub-cell offset, the
points form a uniform n × n grid. Folding the Fourier modes mod n and applying a phase
reduces evaluation to one n × n inverse FFT per offset. In a Δh pair, both meshes see the
same realisation, which is the coupling the estimator needs.

**Reference value ā = I.** The ensemble is invariant under g → −g, so a and 1/a have the
same law. It is also invariant under 90° rotations. In 2D, Keller–Dykhne duality then
gives ā = I exactly. This is the 2D analogue of Lemma 48 for the Gaussian case. All
errors are measured against I.

**Anisotropic test (test 3), exact ā = M.** `--aniso 4 --theta 30` uses
a(x) = s(M^(−1/2) x) · M, where s is the isotropic scalar coefficient above and
M = R(30°) diag(2, ½) R(30°)ᵀ = [[1.625, 0.650], [0.650, 0.875]] (det M = 1). The
substitution x = M^(1/2) y turns the corrector problem for a into the isotropic one for s, so
ā = M^(1/2) · I · M^(1/2) = M exactly, including the off-diagonal entry. The field s(M^(−1/2) x)
has spectral density S(k · M k), sampled on the same dual lattice; det M = 1 keeps unit
variance. The mesh is aligned with neither eigenvector of M. `validate.py --aniso` checks
that a constant s gives exactly c·M, that a laminate converges to its exact (full-tensor)
ā at the h^(2p) rate, and that the field covariance is C_iso(|M^(−1/2) x|). Results are kept
apart from the isotropic ones, in `results/anisotropic/`, `figures/anisotropic/` and
`logs/anisotropic/`.

**FEM** (Definition 9). Periodic uniform meshes with n × n squares of side h = L/n, equispaced
nodes (the standard nodal basis) and (np)² DOFs in both families:

- **P_p on triangles.** Each square is cut along its (1,1) diagonal into two triangles, and each
  triangle carries the total-degree-p Lagrange element, as in the thesis. Integrals use a
  collapsed Gauss rule ((p+1)² points per triangle, exact to degree 2p+1). The matrix has
  17 nonzeros per row for p = 3 (Q3: about 25).
- **Q_p on quadrilaterals.** Tensor-product Lagrange elements with (p+1)² Gauss–Legendre
  points per cell.

The matrix is the stiffness matrix plus (1/T) × the mass matrix. The homogenized coefficient
is computed with the same quadrature (Definition 10.3):
a_ji = L⁻² (δ_ij ∫a − b_j · φ_i), where b_i = −∫ a e_i · ∇η is the load vector.

`validate.py` checks both: a constant coefficient gives exactly c·I, the mass matrix sums to L²,
and for a laminate (exact ā known) the error in ā converges like h^(2p). On triangles the
measured rates are h^1.99, h^3.97, h^5.98 for p = 1, 2, 3 (`figures/fem_convergence_tri.png`).

**Solver.** Unpreconditioned CG, as assumed in Lemma 15. The relative residual tolerance
is 10⁻⁴ (h/L)², i.e. the ε ∼ (h/L)^d of Lemma 15 with a safety factor.

**Estimator** (Definition 13, Eq. 5, Eq. 11). L = 2^s, as Theorem 17 assumes, so that
k_F = 2 log₄ L is an integer. Other parameters:

- h_F = L^(−1/p) and ζ = 2^(1/p); T_k = 4^k, up to T_F = L².
- Levels alternate a Δh step and a ΔT step: Δh at level k compares (T_k, h_k) with (T_k, h_{k+1}); ΔT compares (T_k, h_{k+1}) with (T_{k+1}, h_{k+1}).
- The level-k mesh has n_k = round(L ζ^k) cells per side, because L/h_k must be an integer on the torus.
- Sample sizes follow Eq. (11): N_bc = ⌈L²⌉, N^(h)_k = ⌈L² h_k^(2p)⌉ and N^(T)_k = ⌈L² h_{k+1}^(2p)⌉. With ζ^p = 2 these are L²/4^k and L²/4^(k+1), so the finest ΔT level gets a single sample.

**Measurements.** For each L, the estimator is run R times independently.

- Error: RMS of |a_MC − I|_F over the R runs, with a bootstrap 90% interval. It is split into
  - a statistical part, (Σ_levels V_ℓ / N_ℓ)^½, with each level's variance V_ℓ pooled over all runs (levels with N_ℓ = 1 included);
  - a bias part, |mean a_MC − I|.
- Cost:
  - CPU seconds per estimator run: the sum of the per-sample process times, so it does not depend on how many cores were used;
  - CG work, Σ nnz × iterations × RHS, which does not depend on the machine;
  - the cost model of Lemma 15 and Definition 16, for comparison.

## Results, P_p on triangles

P3 (main case) and P2 (comparison), L = 4, 8, 16, 32 with 128, 64, 32 and 16 independent
estimator runs. Full tables are in `figures/tri_p{3,2}_summary.txt`; logs are in `logs/`.
Brackets are 90% bootstrap intervals (runs resampled within each L, 1000 resamples). The
intervals for the cost slopes only reflect sampling, not machine load, and are very narrow.

| | P3 | P2 | Thm 17 claim |
|---|---|---|---|
| RMS error E[\|a_MC − ā\|²]^½, slope in L | −1.99 [−2.10, −1.90] | −2.01 [−2.18, −1.90] | −2 (up to logs), Eq. (9) |
| statistical part (Σ V_ℓ/N_ℓ)^½ | −1.94 [−1.96, −1.93] | −1.96 [−1.97, −1.95] | |
| bias \|mean a_MC − ā\| | −1.90 | −1.71 | |
| CG work (Σ nnz × iterations × RHS), slope | 4.80 (steps 5.22, 4.66, 4.57) | 4.98 (steps 5.21, 4.98, 4.76) | L⁴ log²L, Eq. (10) |
| CG work / (L⁴ log²L) at L = 4, 8, 16, 32 | 7.1, 7.4, 6.6, 6.2 (×10⁴) | 2.0, 2.1, 2.3, 2.5 (×10⁴) | bounded |
| CPU time per estimator, slope | 4.45 | 4.28 | |

**Accuracy (Eq. 9) holds.** For P3 the RMS error is 1.3e-1, 3.4e-2, 8.3e-3 and 2.2e-3
(slope −1.99, interval [−2.10, −1.90]). The statistical part dominates; the bias is about
half of it. At L = 32 the averaged estimate is I + 8e-4.

**Cost (Eq. 10) holds for P3.** CG work divided by L⁴ log²L stays between 6.2 and 7.4 (×10⁴)
and even falls slightly with L. The local slopes 5.22, 4.66, 4.57 track those of L⁴ log²L
(5.17, 4.83, 4.64), and the Lemma 15 / Definition 16 cost model gives 4.86.
- No level dominates. At L = 32 the CPU shares are base 7%, Δh₀ 20%, Δh₁ 18%, Δh₂ 14%,
  Δh₃ 10%, Δh₄ 5%, and the ΔT terms 8%, 8%, 6%, 3%, 2%.
- Error against CPU time goes like cost^−0.45 (`figures/tri_p3_error_vs_cost.png`).
- **P3 triangles against Q3 quadrilaterals:** at every L the two give the same error within
  the intervals (2.16e-3 against 2.2e-3 at L = 32), while P3 needs about 35% less CG work
  (6.2 against 9.7 ×10⁴ in units of L⁴ log²L at L = 32), mainly from the sparser matrix.

**P2 (outside Theorem 17).** The error still decays like L⁻². CG work divided by L⁴ log²L
grows by 23% from L = 4 to 32, as for Q2 (17%), consistent with the expected L^4.5 log L
growth when p < d + 1.

**Level statistics (P3, L = 32, `figures/tri_p3_level_stats.png`).**
- std(Δh_k) decays like 2^(−1.39k), faster than the h_k^p = 2^(−k) bound of Lemma 18.3.
- std(ΔT_k) decays like 2^(−1.01k), matching T_k^(−1/2) + h_(k+1)^p ∼ 2^(−k) of Lemma 18.4.

### Corrector convergence along the (T, h) path, P3 triangles

`figures/corrector_convergence_tri_p3.png`: same set-up as for Q3 below (L = 4, 8, 16, 32 with
8, 8, 8, 4 realisations, reference two levels past the thesis endpoint).

- **Δh updates do not depend on L.** They shrink by 1.4×, 1.6×, 1.6×, 1.8×, 1.8×, 1.9× per
  level, approaching the ×2 of h^p only slowly (P_p on triangles is slightly more
  pre-asymptotic than Q_p here).
- **ΔT updates** decay at about T^(−1/2) while T ≪ L², then steepen to ×4 per level.
- **At the thesis endpoint**, for L = 4, 8, 16, 32:
  - ‖∇(φ − φ_*)‖ = 3.9e-2, 1.9e-2, 1.1e-2, 5.7e-3, about L^(−0.93);
  - |a − a_*| = 4.9e-3, 2.5e-3, 1.1e-3, 3.3e-4, about L^(−1.3).

### Comparison with standard methods (P3 triangles)

`baselines.py` solves, for 256, 128, 64 and 32 independent coefficient realisations at
L = 4, 8, 16, 32, the corrector along two one-parameter ladders that end at the 2IMC's
finest state (T_F = L², h_F):

- the **h-ladder** (T_F, h_k), k = 0…K: single-index MLMC in the mesh size at fixed T = L²;
- the **T-ladder** (T_k, h_F), k = 0…K: single-index MLMC in the regularisation on the finest mesh;
- the finest state alone gives **plain Monte Carlo**.

All methods therefore have the same bias, and at the same statistical error ε they have the
same RMS error. `compare_methods.py` takes ε² = Σ V_ℓ/N_ℓ of the 2IMC with the Eq. (11) sizes
and computes the cost each method needs to reach it. The baselines get **variance-optimal**
sample sizes N_ℓ ∝ (V_ℓ/C_ℓ)^½, which is generous to them (it needs the level variances
in advance). The 2IMC is shown both with the a priori sizes of Eq. (11) and with optimal sizes.

CG work to reach the 2IMC's accuracy, as a ratio to the 2IMC with Eq. (11) sizes
(90% bootstrap intervals; `figures/tri_p3_methods.txt`, `figures/tri_p3_methods_comparison.png`):

| method | L = 4 | L = 8 | L = 16 | L = 32 | slope in L |
|---|---|---|---|---|---|
| 2IMC, Eq. (11) sizes | 1 | 1 | 1 | 1 | 4.80 |
| 2IMC, optimal sizes | 0.36 | 0.30 [0.28, 0.30] | 0.26 [0.26, 0.27] | 0.25 [0.24, 0.25] | 4.63 |
| MLMC in h only (T = L²) | 0.27 [0.24, 0.29] | 0.23 [0.20, 0.26] | 0.23 [0.19, 0.27] | 0.28 [0.19, 0.34] | 4.81 |
| MLMC in T only (h = h_F) | 0.65 [0.59, 0.74] | 1.20 [1.04, 1.36] | 2.08 [1.72, 2.38] | 3.71 [2.65, 4.43] | 5.64 |
| plain Monte Carlo | 0.62 [0.54, 0.73] | 1.33 [1.08, 1.55] | 2.70 [2.14, 3.24] | 7.48 [4.85, 9.38] | 5.98 |

CPU time gives the same picture (ratios at L = 32: 0.25, 0.30, 3.96, 6.24).

- **Against plain Monte Carlo and MLMC in T only, the 2IMC wins, by a margin that grows with L.**
  Those two scale like L^5.6–6.0 and need 3.7× and 7.5× the work of the 2IMC at L = 32.
  Both solve every sample on the finest mesh.
- **Against MLMC in h only, the 2IMC is on par in this range.** With optimal sizes the 2IMC
  is ahead at L = 16 and 32 (0.25 against 0.28 at L = 32) and has the smaller slope (4.63
  against 4.81), but the intervals overlap. With the Eq. (11) sizes it is about 3.5× more
  expensive than MLMC in h only.
- **Why MLMC in h only does well here.** For this smooth field (ν = 4.5) and P3 elements, the
  mesh corrections are tiny: at L = 32, Var Δh₀ = 2.7e-8 against 3.0e-3 for the base level.
  Almost all of MLMC-in-h's work is its coarsest level, a plain Monte Carlo at (T = L², h = 1).
  The 2IMC's base level at T = 1 is about 5.5× cheaper per sample (146 against 809 CG
  iterations at L = 32), but the regularisation has to be removed again by the ΔT terms.
  Their variance falls about as fast as their cost grows (V·C is within a factor 4 across
  the first four ΔT levels), so with optimal sizes the ΔT levels together cost about 1.3×
  the base level; this is the regime that produces the log factors of Theorem 17. Overall
  the two effects nearly cancel at L ≤ 32. When the mesh corrections are not negligible, the
  2IMC pulls ahead (see the rough-field comparison below).

### Sample sizes: Eq. (11) against the variance-optimal choice (P3 triangles)

`figures/tri_p3_sample_sizes.png` compares, at L = 32, the Eq. (11) sizes with the
variance-optimal sizes that reach the same statistical error from the measured level
variances and costs:

| term | base | Δh₀ | ΔT₀ | Δh₁ | ΔT₁ | Δh₂ | ΔT₂ | Δh₃ | ΔT₃ | Δh₄ | ΔT₄ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Eq. (11) | 1024 | 1024 | 256 | 256 | 64 | 64 | 16 | 16 | 4 | 4 | 1 |
| optimal | 1684 | 3 | 59 | 1 | 23 | 1 | 9 | 1 | 3 | 1 | 1 |

The optimal allocation costs 0.25× (L = 32) to 0.36× (L = 4) of Eq. (11). Eq. (11) puts too
few samples on the base level and far too many on the Δh levels. It is derived from the
a priori bound std(Δh_k) ≲ h_k^p (Lemma 18.3), whose rate is right but whose constant is
orders of magnitude pessimistic for this field. The ΔT sizes are within a factor 2–4 of
optimal. Since the optimal sizes keep the same L-scaling (slope 4.63), this changes the
constant, not Theorem 17. In practice a pilot run to estimate V_ℓ, as in standard MLMC,
would recover the factor 3–4.

### Robustness in contrast, correlation length and smoothness (P3 triangles)

Seven variants of the coefficient field, each with L = 4, 8, 16, 32 and 128, 64, 32, 16 runs
(`figures/tri_p3_robustness.{png,txt}`; the baseline row is the main run). ā = I still holds
exactly for every variant, since the symmetry argument does not depend on β, ℓ or ν.

| configuration | RMSE slope [90% CI] | stat. sd slope | RMSE at L = 32 | work/(L⁴ log²L) at L = 4, 8, 16, 32 (×10⁴) |
|---|---|---|---|---|
| baseline (β = 1, ℓ = 0.25, ν = 4.5) | −1.99 [−2.10, −1.90] | −1.94 | 2.2e-3 | 7.1, 7.4, 6.6, 6.2 |
| β = 2 (contrast e⁴ ≈ 55) | −1.86 [−1.96, −1.78] | −1.86 | 9.3e-3 | 10.8, 12.1, 10.9, 10.4 |
| β = 3 (contrast e⁶ ≈ 400) | −1.83 [−1.91, −1.76] | −1.81 | 2.6e-2 | 16.2, 19.8, 18.0, 17.2 |
| ℓ = 0.125 | −1.84 [−1.97, −1.74] | −1.99 | 1.3e-3 | 6.9, 6.9, 6.1, 5.8 |
| ℓ = 0.5 | −1.88 [−1.99, −1.79] | −1.86 | 6.5e-3 | 5.4, 6.9, 6.8, 6.7 |
| ν = 2.5 (violates ν > p + 1) | −1.94 [−2.12, −1.81] | −1.96 | 1.6e-3 | 7.1, 7.3, 6.4, 6.1 |
| ν = 1.5 (rough, cut at \|k\| ≤ 75) | −1.54 [−1.63, −1.46] | −1.97 | 3.3e-3 | 6.9, 7.0, 6.2, 5.9 |
| ν = 0.5 (very rough, cut at \|k\| ≤ 75) | −0.46 [−0.49, −0.43] | −1.85 | 3.5e-2 | 6.2, 6.2, 5.6, 5.4 |

- **Cost is robust.** CG work divided by L⁴ log²L stays flat or falls slightly up to L = 32 in
  every variant (CG-work slopes 4.78–4.97). Higher contrast raises the constant (1.7× for
  β = 2, 2.8× for β = 3 at L = 32, from the worse conditioning) but not the rate.
- **Contrast and correlation length change constants, and the rate only by a slowly growing
  factor.** Larger contrast or a longer correlation length means fewer effectively independent
  cells per box, so larger errors, and a larger bias share (at β = 3 the bias, 2.1e-2, exceeds
  the statistical error, 1.4e-2, at every L). The RMSE slopes stay between −1.83 and −1.88
  with L = 32 included, so the shortfall from −2 is not shrinking. For β it sits in the
  statistical part (slopes −1.86, −1.81): the ΔT terms take a growing share of the variance
  (β = 3: 52% at L = 16, 67% at L = 32; baseline: 16% and 21%). Each ΔT level contributes
  about equally under Eq. (11), so their total grows with the number of levels, ∼log L. This
  is the (log L)^C of Eq. (9) becoming visible, not a different power.
- **Smoothness: the condition ν > p + 1 of Definition 12 is sufficient but not sharp.** ν = 2.5
  behaves like the baseline up to L = 32. At ν = 1.5 the statistical part still decays like
  L⁻² (slope −1.97), but the bias decays only like L^(−1.15) (3.4e-2, 1.3e-2, 6.3e-3, 3.0e-3),
  and at L = 32 it is three times the statistical error. The RMSE slope therefore drops from
  −1.74 (L ≤ 16) to −1.54 (L ≤ 32) and will approach the bias rate for larger L. At ν = 0.5
  the bias stagnates (8.1e-2 to 3.5e-2, slope −0.4) and dominates: the coefficient varies on
  scales (down to 2π/75 ≈ 0.08) far below the finest mesh h_F = L^(−1/3) ≈ 0.31 at L = 32, so
  the FEM does not resolve it. The statistical part (slope −1.85) is unaffected, so the loss is
  entirely in the discretisation bias, which is the part the smoothness assumption controls.
- **Mesh corrections matter for rough fields.** The Δh level variances (right panel, L = 32)
  grow by one to two orders of magnitude as ν decreases. With the Eq. (11) sizes the Δh levels
  contribute under 0.5% of the estimator's statistical variance for every smooth variant,
  1.6% for ν = 1.5 and 51% for ν = 0.5 (L = 32; at L = 16: 1.4% and 40%). Accordingly the
  optimal allocation saves less: from the 2IMC's own level statistics it costs 0.20× of
  Eq. (11) for the baseline, 0.24× for ν = 1.5, 0.54× for ν = 0.5 and 0.28×, 0.31× for
  β = 2, 3, the same at L = 16 and 32. (`compare_methods.py`, which rounds the sizes to whole
  samples, gives 0.25×, 0.27× and 0.58× for the first three at L = 32.)

### Comparison with standard methods for rough fields (P3 triangles)

The method comparison repeated for ν = 1.5 and ν = 0.5 (L = 4, 8, 16, 32; baselines with 256,
128, 64, 32 realisations; figures `figures/tri_p3_nu{1.5,0.5}_kmax75_methods*`).
CG work to reach the 2IMC's accuracy, as a ratio to the 2IMC with Eq. (11) sizes, at L = 32
(L = 16 in parentheses):

| method | baseline ν = 4.5 | ν = 1.5 | ν = 0.5 | slope in L, ν = 1.5 / 0.5 |
|---|---|---|---|---|
| 2IMC, Eq. (11) sizes | 1 | 1 | 1 | 4.79 / 4.80 |
| 2IMC, optimal sizes | 0.25 [0.24, 0.25] (0.26) | 0.27 [0.27, 0.28] (0.30) | 0.58 [0.57, 0.60] (0.58) | 4.66 / 4.82 |
| MLMC in h only | 0.28 [0.19, 0.34] (0.23) | 0.38 [0.27, 0.47] (0.39) | 1.00 [0.80, 1.15] (0.99) | 4.93 / 5.15 |
| MLMC in T only | 3.71 [2.65, 4.43] (2.08) | 2.69 [1.83, 3.39] (2.16) | 1.10 [0.73, 1.43] (1.19) | 5.47 / 5.10 |
| plain Monte Carlo | 7.48 [4.85, 9.38] (2.70) | 6.66 [4.12, 9.07] (4.17) | 3.54 [2.15, 4.82] (2.59) | 5.94 / 5.71 |

- **For rough fields the 2IMC is the cheapest method, and L = 32 confirms it.** With optimal
  sizes it needs 0.71× (ν = 1.5) and 0.58× (ν = 0.5) of the work of MLMC in h only at
  L = 32 (0.77× and 0.59× at L = 16), and its cost grows more slowly over L = 4–32 (L^4.66
  against L^4.93, and L^4.82 against L^5.15). At ν = 0.5 even the a priori Eq. (11) sizes
  match MLMC in h with optimal sizes.
- **How firm this is.** At ν = 0.5 the lead is clear: the MLMC-in-h interval [0.80, 1.15]
  lies well above the 2IMC's 0.58. At ν = 1.5 the lower end of the MLMC-in-h interval (0.27)
  touches the 2IMC's value, so the lead is likely but not certain at the 90% level. The work
  ratio to MLMC in h has fallen steadily with L at ν = 1.5 (1.28, 0.94, 0.77, 0.71); at
  ν = 0.5 it fell from L = 4 to 16 (1.14, 0.75, 0.59) and stayed at 0.58 at L = 32.
- The reason is the one given above, now in the other direction. Once the mesh corrections
  carry real variance, MLMC in h has to pay for them at T = L², where every solve needs many
  CG iterations, while the 2IMC takes its Δh steps at small T, where solves are cheap.
- In CPU time (more affected by machine load) the ordering is the same at L = 32: 2IMC
  optimal 0.30 against 0.39 for MLMC in h at ν = 1.5, and 0.60 against 0.99 at ν = 0.5.
- Caveat: at ν ≤ 1.5 the RMS error at L = 32 is dominated by the discretisation bias, which is
  the same for all methods (they share the finest state); the comparison is at equal
  statistical error.


## Results, Q_p on quadrilaterals

Q3 (main case) and Q2 (comparison), L = 4, 8, 16, 32 with 128, 64, 32 and 16 independent
estimator runs. Full tables are in `figures/p{3,2}_summary.txt`; logs are in `logs/`.

| | Q3 | Q2 | Thm 17 claim |
|---|---|---|---|
| RMS error E[\|a_MC − ā\|²]^½, slope in L | −1.99 | −2.02 | −2 (up to logs), Eq. (9) |
| statistical part (Σ V_ℓ/N_ℓ)^½ | −1.94 | −1.96 | |
| bias \|mean a_MC − ā\| | −1.91 | −1.72 | |
| CG work (Σ nnz × iterations × RHS), slope | 4.83 (steps 5.29, 4.69, 4.56) | 4.96 (steps 5.18, 4.94, 4.75) | L⁴ log²L, Eq. (10) |
| CG work / (L⁴ log²L) at L = 4, 8, 16, 32 | 1.04, 1.13, 1.03, 0.97 (×10⁵) | 2.49, 2.51, 2.71, 2.92 (×10⁴) | bounded |
| CPU time per estimator, slope | 4.68 | 4.35 | |

**Accuracy (Eq. 9) holds.** For Q3 the RMS error is 1.3e-1, 3.4e-2, 8.2e-3 and 2.2e-3,
a slope of −1.99. The statistical part dominates, and the bias is about half of it. At L = 32
the averaged estimate is I + 8e-4, within two standard errors of the exact answer.

**Cost (Eq. 10) holds for Q3.** CG work divided by L⁴ log²L stays between 0.97 and 1.13 (×10⁵)
across L = 4–32, and the local slopes 5.29, 4.69, 4.56 track those of L⁴ log²L
(5.17, 4.83, 4.64). The cost model of Lemma 15 and Definition 16 gives slope 4.86.
- No level dominates any more. At L = 32 the CPU share per term decreases slowly from the
  coarse to the fine levels (Δh₀ 19%, Δh₁ 20%, Δh₂ 14%, Δh₃ 9%, Δh₄ 5%; the single finest ΔT
  sample 1.6%). Under the draft's sample sizes the two finest terms took about 45%.
- Error against CPU time goes like cost^−0.43, consistent with L⁻² against L⁴ log²L over this
  range of L; the cost^−1/2 Monte Carlo rate is reached only up to the log² factor.
- The CPU-time slope (last step 4.88) is noisier than the CG work, because the machine was
  heavily loaded by other programs during the L = 32 run (load average ≈ 550 on 8 cores).
  The CG-work column does not depend on the machine.
- Compared with the draft's sample sizes, one Q3 estimator at L = 32 is about 5.5× cheaper
  in CG work.

**Q2 (outside Theorem 17).** The error still decays like L⁻². The CG work divided by
L⁴ log²L creeps up by 17% from L = 4 to 32, consistent with the expected L^4.5 log L growth
when p < d + 1, but at L ≤ 32 the two rates cannot be told apart clearly.

**Level statistics (Q3, L = 32, `figures/p3_level_stats.png`).**
- std(Δh_k) decays like 2^(−1.61k), faster than the h_k^p = 2^(−k) bound of Lemma 18.3.
- std(ΔT_k) decays like 2^(−1.01k), matching T_k^(−1/2) + h_(k+1)^p ∼ 2^(−k) of Lemma 18.4.

### Corrector convergence along the (T, h) path, Q3

`figures/corrector_convergence_p3.png` uses Q3, the estimator's CG solver and tolerance, and
L = 4, 8, 16, 32 with 8, 8, 8 and 4 realisations. Each realisation follows
(1, 1) → (1, h₁) → (4, h₁) → … → (L², h_F) and then two more levels, whose last state is the
reference φ_*. Correctors are compared in the energy norm (L⁻² ∫ Σᵢ |∇uᵢ|²)^½.

- **Δh updates do not depend on L.** The curves for all L coincide. They shrink by 1.5×, 1.7×,
  1.7×, 1.9×, 1.9×, 1.9× per level, approaching the ×2 of h^p = 2^(−k) only on the finer meshes.
- **ΔT updates depend on L.** They decay at about the T^(−1/2) rate (×2 per level) while T is
  small compared with L², then steepen towards ×4 per level (like L²/T) once the Poincaré
  constant of the torus takes over from the massive term.
- **At the thesis endpoint** (T = L², h = h_F), for L = 4, 8, 16, 32:
  - ‖∇(φ − φ_*)‖ = 3.4e-2, 1.4e-2, 7.8e-3, 4.0e-3, about L^(−1.0), matching the h_F^p = L⁻¹
    energy error;
  - |a − a_*| = 4.6e-3, 2.5e-3, 1.1e-3, 3.2e-4, about L^(−1.3).

  Both are lower bounds on the true error, since φ_* is only two levels further along.

## Earlier runs (draft thesis, superseded)

`results/archive_thesis_v4/` and `figures/archive_thesis_v4/` hold the first runs. They used
the draft's sample sizes N_k = ⌈L² h_k^(p−δ)⌉ (δ = 0.1) and its condition p ≥ 2. Under those
sizes:
- The error still decayed like L⁻².
- The cost grew like L^5.6 (Q2) and L^5.1–5.4 (Q3). The draft's cost series Σ(2ζ^(δ+d+1−p))^k
  has ratio > 1, so the finest level dominated.
- The final thesis fixes this by switching to h_k^(2p) sample sizes and requiring p ≥ d + 1.
  The series then has ratio ≤ 1, giving L⁴ log²L.

The original write-up of those runs follows, with the draft's numbering (its Theorem 16 is
Theorem 17 of the final version, and its Lemma 17 is Lemma 18).

### Results (Q2: L = 4–32 with 256/128/32/8 runs; Q3: L = 4–32 with 128/64/16/2 runs)

Full tables are in `figures/archive_thesis_v4/p{2,3}_summary.txt`.

| | Q2 slope in L | Q3 slope in L | Thm 16 claim |
|---|---|---|---|
| RMS error E[\|a_MC − ā\|²]^½ | −2.08 | −1.90 (L=32: only 2 runs) | −2 (up to logs), Eq. (9) |
| statistical part (Σ Var/N)^½ | −2.00 | −1.99 | |
| CG work (Σ nnz × iterations) | 5.67 (last step 5.56) | 5.37 (last step 5.14) | 4 (+ log²), Eq. (10) |
| CPU time | 5.33 | 5.12 | |

- **The accuracy claim (9) holds.** The error decays like L⁻² for both degrees. The bias is about half the statistical error, and |E a_MC − ā| at L = 4, 8, 16 is 4.9e-2, 1.1e-2, 3.9e-3. At L = 32 the averaged estimate is I + 5e-4, which is within noise of the exact answer.
- **The cost claim (10) does not hold as stated.** Work grows like L^5.6 (Q2) and L^5.1–5.4 (Q3). The likely cause is in the proof of Eq. (10). The series Σ_k (2ζ^(δ+d+1−p))^k has ratio 2ζ^(δ+d+1−p) > 1 for every p when ζ = 2^(1/p). It is therefore dominated by its last term, not ≲ log L, which gives K_tot ∼ L^(d+2+(d+1+δ)/p) log L. That is 5.55 for p = 2 and 5.03 for p = 3, matching the measured last-step slopes. The cost-breakdown figure shows the same thing: the two finest terms take about 45% of the CPU, and the base level takes 0.4%.
- Consequently the error scales like cost^−0.39 (Q2), not cost^−1/2. The predicted exponent −2/(4 + (3+δ)/p) tends to −1/2 only as p → ∞.
- **Level statistics (L = 32, Q2).**
  - std(Δh_k) decays like 2^(−1.72k), faster than the h_k^p = 2^(−k) of Lemma 17(3).
  - std(ΔT_k) decays like 2^(−0.98k), matching the T_k^(−1/2) of Lemma 17(4).
  - The ΔT means are about 5× the Δh means, so the regularisation steps carry most of the telescoped correction.
- **Caveat.** Other heavy load was on the machine during the Q2 L=32 run, so its CPU time is inflated. The CG-work column does not depend on the machine and is the cleaner cost measure.

### Corrector convergence along the (T, h) path

`figures/archive_thesis_v4/corrector_convergence_p2.png` uses Q2, the estimator's CG solver and tolerance, and
L = 4, 8, 16, 32 with 8, 8, 8 and 4 realisations.

Each coefficient realisation is followed along (1, 1) → (1, h₁) → (4, h₁) → … → (L², h_F). The
path then continues two more levels; that last state is the reference φ_*. Correctors are
compared in the energy norm (L⁻² ∫ Σᵢ |∇uᵢ|²)^½ at the reference mesh's quadrature points.
A 10⁴× tighter CG tolerance changes no number to 4 digits.

- **Δh steps do not depend on L.** The curves for all four L coincide, as expected for a local
  discretisation error. Each update shrinks by 1.5×, 1.7×, 1.8×, … per level, approaching the
  asymptotic ×2 of h^p only around h ≈ 0.25. The meshes on the thesis path are pre-asymptotic
  relative to the field's correlation scale.
- **ΔT steps depend on L.**
  - While T < L², they decay at about the T^(−1/2) rate of Lemma 17(4).
  - Once T > L², the decay steepens to about ×4 per level, i.e. like L²/T, because the
    Poincaré constant takes over from the massive term.
  - The thesis endpoint T_F = L² sits at this knee.
- **At the thesis endpoint**, for L = 4, 8, 16, 32:
  - ‖∇(φ − φ_*)‖ = 6.0e-2, 3.5e-2, 1.7e-2, 9.0e-3, about L^(−0.9). This is consistent with the
    h_F^p = L⁻¹ energy error.
  - |a − a_*| = 6.6e-3, 2.4e-3, 1.0e-3, 3.3e-4, about L^(−1.4).

  Both are lower bounds on the true error, since φ_* is itself only two levels further along.
  The sharp final drops in the bottom panels are this artefact of the reference being close.
