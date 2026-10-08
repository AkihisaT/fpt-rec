#!/bin/bash
export OMP_NUM_THREADS=2
for s in -1 0; do
  python blis_df.py $s 2 - > blis_df_s${s}.out 2>&1
  python blis_df.py $s 2 0,9,18,27,36 > blis_df_s${s}_ho.out 2>&1
done
echo blis df done
