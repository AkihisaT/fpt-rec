#!/bin/bash
# 32a aff2: BF + DF set (rings 0-2 + 6-7, d32a/work_df_aff2). Starts after the BF BLIS runs of chainF2.
export OMP_NUM_THREADS=4
R=$PWD
until grep -q "blisbf done" d32a/runs/chainF2_bfaff2.log 2>/dev/null; do sleep 30; done
cd $R/d32a && mkdir -p blis_dfaff2
for c in 0 1; do
  python -c "import numpy as np; z=np.load('blis_bfaff2/c${c}_ho.npz'); np.savez('blis_dfaff2/init_from_bf_c${c}.npz', a=z['a'], phi=z['phi'], kappa_solver=z['kappa'])"
  python blis_32a.py --work work_df_aff2 --dr 0.25 --c $c --exclude val9 --init blis_dfaff2/init_from_bf_c${c}.npz --out blis_dfaff2/c${c}_ho.npz > blis_dfaff2/c${c}_ho.log 2>&1
done
echo blisdf done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32adfaff2.json --steps train --fov-factors $c > log_32adfaff2_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32adfaff2_it30.json --steps train --fov-factors $c > log_32adfaff2_it30_c$c.txt 2>&1; done
cd $R/epry_pipeline
python run_epry.py --config config_epry_32adfaff2.json --steps train > log_32adfaff2.txt 2>&1
python run_epry.py --config config_epry_32adfaff2_1000.json --steps train > log_32adfaff2_1000.txt 2>&1
echo df512 done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32adfaff2_1000.json --steps train --fov-factors $c > log_32adfaff2_1000_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32adfaff2_1000_it30.json --steps train --fov-factors $c > log_32adfaff2_1000_it30_c$c.txt 2>&1; done
echo chainG2 done
