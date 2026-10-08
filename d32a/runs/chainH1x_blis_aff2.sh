#!/bin/bash
# split-half BLIS-FPM (FOV c = 1): BF halves from zero, BF+DF halves warm-started from the BF half of the same set
export OMP_NUM_THREADS=3
cd "$(dirname "$0")/.."
for h in A B; do
  python blis_32a.py --work ../fpt_pipeline/fpt_output_32a_bf_aff2/work --dr 0.25 --c 1 --exclude $(cat frc/ex_bf$h.txt) --init zero --threads 3 --out blis_halves2/bf${h}_c1.npz > blis_halves2/bf${h}_c1.log 2>&1
done
for h in A B; do
  python -c "import numpy as np; z=np.load('blis_halves2/bf${h}_c1.npz'); np.savez('blis_halves2/init_df${h}.npz', a=z['a'], phi=z['phi'], kappa_solver=z['kappa'])"
  python blis_32a.py --work work_df_aff2 --dr 0.25 --c 1 --exclude $(cat frc/ex_df$h.txt) --init blis_halves2/init_df${h}.npz --threads 3 --out blis_halves2/df${h}_c1.npz > blis_halves2/df${h}_c1.log 2>&1
done
echo "chainH1x done"
