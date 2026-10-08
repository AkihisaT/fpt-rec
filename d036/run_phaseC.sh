#!/bin/bash
# 036 Phase C (rings 1 + 3): EPRY 512, BLIS-FPM (full / hold-out, plane / FOV), then DIP 512 (after the Phase-B 1024 DIP chain)
W=$(cd "$(dirname "$0")/.." && pwd); cd $W
export OMP_NUM_THREADS=2
(cd epry_pipeline && python run_epry.py --config config_epry_036df.json --steps train --threads 2 > run036df_epry.out 2>&1)
while [ ! -f d036/blis_df/s-1_ho.json ]; do sleep 20; done
export BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$W/d036/blis_df13
for s in -1 0; do
  (cd d036 && python blis_df.py $s 2 - > blis_df13_s${s}.out 2>&1)
  (cd d036 && python blis_df.py $s 2 0,25,34,43 > blis_df13_s${s}_ho.out 2>&1)
done
echo "BLIS13 done"
# halves (FRC): FOV model, bright only (ring 1) and bright + dark (rings 1, 3)
for rings in 1 1,3; do
  for h in A B; do
    (cd d036 && BLIS_DF_RINGS=$rings BLIS_DF_DATA=$h BLIS_DF_OUT=$W/d036/blis_half_r${rings/,/} python blis_df.py -1 2 - > blis_half_r${rings/,/}_$h.out 2>&1)
  done
done
echo "halves done"
