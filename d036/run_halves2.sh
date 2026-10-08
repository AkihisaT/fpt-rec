#!/bin/bash
W=$(cd "$(dirname "$0")/.." && pwd); cd $W/d036
export OMP_NUM_THREADS=2
for rings in 1 1,3; do
  for h in A B; do
    BLIS_DF_RINGS=$rings BLIS_DF_DATA=$h BLIS_DF_OUT=$W/d036/blis_half_r${rings/,/} python blis_df.py -1 2 - > blis_half_r${rings/,/}_$h.out 2>&1 &
  done
  wait
done
echo "halves2 done"
