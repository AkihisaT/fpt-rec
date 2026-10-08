#!/bin/bash
# 32a rings 0-2 + dark-field rings 6-7 (155 images, d32a/work_df_nd): BLIS-FPM (18 held out), DIP 512 (600 it, 30 it),
# EPRY 512 / 1000, DIP 1000 (300 it, 30 it). Starts after chainC. Run from the workspace root.
export OMP_NUM_THREADS=4
R=$PWD
until grep -q "chainC done" d32a/runs/chainC_bf.log; do sleep 30; done
cd $R/d32a && mkdir -p blis_df && for c in 0 1; do python blis_32a.py --work work_df_nd --dr 0.25 --c $c --exclude val9 --out blis_df/c${c}_ho.npz > blis_df/c${c}_ho.log 2>&1; done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32adf.json --steps train --fov-factors $c > log_32adf_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32adf_it30.json --steps train --fov-factors $c > log_32adf_it30_c$c.txt 2>&1; done
cd $R/epry_pipeline
python run_epry.py --config config_epry_32adf.json --steps train > log_32adf.txt 2>&1
python run_epry.py --config config_epry_32adf_1000.json --steps train > log_32adf_1000.txt 2>&1
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32adf_1000.json --steps train --fov-factors $c > log_32adf_1000_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32adf_1000_it30.json --steps train --fov-factors $c > log_32adf_1000_it30_c$c.txt 2>&1; done
echo chainE done
