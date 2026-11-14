#!/bin/zsh
# L = 32 for the rough-field comparison and the robustness study (most informative first).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
B="--element P --p 3 --L 4 8 16 32 --reals 256 128 64 32"   # reproduces the L <= 16 realisations, adds L = 32
$PY baselines.py ${=B} --nu 1.5 --kmax 75 > logs/baselines_tri_p3_nu1.5.log 2>&1
$PY baselines.py ${=B} --nu 0.5 --kmax 75 > logs/baselines_tri_p3_nu0.5.log 2>&1
R="--element P --p 3 --L 32 --reps 16"                      # L <= 16 already exist and are skipped
$PY run_experiments.py ${=R} --nu 1.5 --kmax 75 >> logs/robust_tri_p3_nu1.5.log 2>&1
$PY run_experiments.py ${=R} --nu 0.5 --kmax 75 >> logs/robust_tri_p3_nu0.5.log 2>&1
$PY run_experiments.py ${=R} --nu 2.5           >> logs/robust_tri_p3_nu2.5.log 2>&1
$PY run_experiments.py ${=R} --ell 0.125        >> logs/robust_tri_p3_ell0.125.log 2>&1
$PY run_experiments.py ${=R} --ell 0.5          >> logs/robust_tri_p3_ell0.5.log 2>&1
$PY run_experiments.py ${=R} --beta 2           >> logs/robust_tri_p3_beta2.log 2>&1
$PY run_experiments.py ${=R} --beta 3           >> logs/robust_tri_p3_beta3.log 2>&1
echo queue3 done
