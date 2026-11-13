# Numerics for the two-index Monte Carlo (2IMC) estimator, d = 2

Checks Theorem 17 of the thesis (B. C. White, *Two-Index Monte Carlo Method for
Approximating Coefficients of Linear Elliptic Problems*, ISTA MSc thesis 2026, final version):
the error bound E|a_MC − ā| ≲ L⁻² (log L)^C (Eq. 9) and the cost bound K_tot ≲ L⁴ log²L
(Eq. 10). In d = 2 the theorem needs p ≥ d + 1 = 3, so Q3 is the main case. Q2 is run for
comparison. Theorem, lemma and definition numbers below follow the final version.

## Files

| file | contents |
|---|---|
| `twoimc.py` | the Gaussian field, the periodic Q_p FEM, CG, the level hierarchy and one-sample routines |
| `validate.py` | checks of the field and the FEM (`--plot` draws `figures/fem_convergence.png`) |
| `run_experiments.py` | runs R independent copies of the estimator for each L, in parallel; writes `results/p{p}_L{L}.{npz,json}`. It checkpoints every 5 minutes, so rerunning the same command after an interruption resumes, and finished L values are skipped |
| `corrector_convergence.py`, `plot_corrector.py` | Cauchy convergence of the corrector along the (T, h) path for fixed L; writes `results/corrector_p{p}.json` and `figures/corrector_convergence_p{p}.{png,txt}` |
| `make_plots.py` | analysis, the figures in `figures/`, and `figures/p{p}_summary.txt` |

```
python -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
.venv/bin/python validate.py --plot
.venv/bin/python run_experiments.py --p 3 --L 4 8 16 32 --reps 128 64 32 16
.venv/bin/python run_experiments.py --p 2 --L 4 8 16 32 --reps 128 64 32 16
.venv/bin/python make_plots.py
.venv/bin/python corrector_convergence.py --p 3 --L 4 8 16 32 --reals 8 8 8 4 --extra 2
.venv/bin/python plot_corrector.py --p 3
```

## What is implemented

**Coefficient field** (Definition 12, Assumption 4). g is a centred stationary Gaussian field on the
torus [0, L)², built from a Fourier series with Matérn-type spectral density
S(k) ∝ (1 + ℓ²|k|²)^−(ν+1), with ν = p + 1.5 (Definition 12 needs ν > p + 1), ℓ = 0.25
(correlation ≈ 0.29 at |x| = 1) and Var g = 1. Sampling S on the dual lattice 2πℤ²/L is the
same as periodising the whole-space covariance, so the ensemble does not depend on L. This is
the periodisation a_L that Assumption 4 leaves unspecified.
Modes carrying the last 10⁻⁶ of the variance are dropped. The coefficient is
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

**FEM** (Definition 9). Q_p Lagrange elements on the uniform periodic n × n quadrilateral
mesh, with equispaced nodes (the standard nodal basis); there are (np)² DOFs. The thesis
analyses total-degree-p elements on simplices; quadrilaterals were chosen for this implementation. The matrix is
the stiffness matrix plus (1/T) × the mass matrix. All integrals use (p+1)² Gauss–Legendre
points per cell. The homogenized coefficient is computed with the same quadrature
(Definition 10.3):
a_ji = L⁻² (δ_ij ∫a − b_j · φ_i), where b_i = −∫ a e_i · ∇η is the load vector.

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

## Results (final thesis)

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

### Corrector convergence along the (T, h) path (Q3)

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
