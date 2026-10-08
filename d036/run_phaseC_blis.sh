#!/bin/bash
W=$(cd "$(dirname "$0")/.." && pwd); cd $W/d036
export OMP_NUM_THREADS=2
for s in -1 0; do
  BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$W/d036/blis_df13 python blis_df.py $s 2 - > blis_df13_s${s}.out 2>&1
  BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$W/d036/blis_df13 python blis_df.py $s 2 0,25,34,43 > blis_df13_s${s}_ho.out 2>&1
done
echo "BLIS13 done"
for rings in 1 1,3; do
  for h in A B; do
    BLIS_DF_RINGS=$rings BLIS_DF_DATA=$h BLIS_DF_OUT=$W/d036/blis_half_r${rings/,/} python blis_df.py -1 2 - > blis_half_r${rings/,/}_$h.out 2>&1
  done
done
echo "halves done"
