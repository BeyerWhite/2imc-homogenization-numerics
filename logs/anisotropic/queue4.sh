#!/bin/zsh
# Test 3 (anisotropy), after queue3. Results kept apart from the isotropic ones: results/anisotropic/.
cd "$(dirname "$0")/../.."
until grep -q "queue3 done" logs/queue3.out 2>/dev/null; do sleep 120; done
.venv/bin/python run_experiments.py --element P --p 3 --L 4 8 16 32 --reps 128 64 32 16 \
    --aniso 4 --theta 30 --out results/anisotropic > logs/anisotropic/run_tri_p3_aniso4_theta30.log 2>&1
echo queue4 done
