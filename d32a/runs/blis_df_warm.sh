#!/bin/bash
# BLIS-FPM dark-field set, warm start from the bright-field held-out BLIS objects (same 7 bright held-out images).
cd d32a
until grep -q "done:" blis_df/c0_ho.log; do sleep 20; done
OMP_NUM_THREADS=2 python blis_32a.py --work work_df_nd --dr 0.25 --c 1 --exclude val9 --init blis_df/init_from_bf_c1.npz --out blis_df/c1_ho.npz > blis_df/c1_ho.log 2>&1
echo blisdfwarm done
