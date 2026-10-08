#!/bin/bash
W=$(cd "$(dirname "$0")/.." && pwd); cd $W/d036
export OMP_NUM_THREADS=2 BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$W/d036/blis_k3test
for p in 0.98 1.06; do
  BLIS_DF_PUPIL=$p python blis_df.py -1 2 0,25,34,43 100 > blis_ptest_$p.out 2>&1 &
done
wait; echo "ptest done"
