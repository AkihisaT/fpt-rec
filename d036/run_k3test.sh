#!/bin/bash
W=$(cd "$(dirname "$0")/.." && pwd); cd $W/d036
export OMP_NUM_THREADS=2 BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$W/d036/blis_k3test
for f in 1 0.97 1.03; do
  BLIS_DF_KSCALE3=$f python blis_df.py -1 2 0,25,34,43 100 > blis_k3test_$f.out 2>&1 &
done
wait; echo "k3 done"
