#!/bin/zsh
# Second batch: waits for the triangle reruns (corrector_tri_p3.json is their last output).
cd "$(dirname "$0")/.."
while [ ! -f results/corrector_tri_p3.json ]; do sleep 120; done
PY=.venv/bin/python
$PY baselines.py --element P --p 3 --L 4 8 16 32 --reals 256 128 64 32 > logs/baselines_tri_p3.log 2>&1
R="--element P --p 3 --L 4 8 16 --reps 128 64 32"
$PY run_experiments.py ${=R} --beta 2          > logs/robust_tri_p3_beta2.log 2>&1
$PY run_experiments.py ${=R} --beta 3          > logs/robust_tri_p3_beta3.log 2>&1
$PY run_experiments.py ${=R} --ell 0.125       > logs/robust_tri_p3_ell0.125.log 2>&1
$PY run_experiments.py ${=R} --ell 0.5         > logs/robust_tri_p3_ell0.5.log 2>&1
$PY run_experiments.py ${=R} --nu 2.5          > logs/robust_tri_p3_nu2.5.log 2>&1
$PY run_experiments.py ${=R} --nu 1.5 --kmax 75 > logs/robust_tri_p3_nu1.5.log 2>&1
$PY run_experiments.py ${=R} --nu 0.5 --kmax 75 > logs/robust_tri_p3_nu0.5.log 2>&1
echo queue2 done
