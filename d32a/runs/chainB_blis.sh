#!/bin/bash
# 32a BLIS-FPM: FOV-strength scan (bright field, 7 held out) and dark-field registration test. Run from the workspace root.
export OMP_NUM_THREADS=4
cd d32a; mkdir -p fov_scan blis_dftest
W=../fpt_pipeline/fpt_output_32a_bf_reg2/work
for c in 0.5 1.25 1.5 2.0; do python blis_32a.py --work $W --dr 0.25 --c $c --exclude val9 --out fov_scan/c$c.npz > fov_scan/c$c.log 2>&1; done
for t in df_reg2 df_nd; do python blis_32a.py --work work_$t --dr 0.25 --c 1 --exclude val9 --out blis_dftest/$t.npz > blis_dftest/$t.log 2>&1; done
echo chainB done
