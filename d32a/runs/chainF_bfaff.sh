#!/bin/bash
# 32a after the posaffine step (mechanical drift removed): BF set (rings 0-2). Run from the workspace root.
export OMP_NUM_THREADS=4
R=$PWD
while pgrep -f "config_32a_bf_aff.json" > /dev/null; do sleep 20; done
python d32a/make_df_aff_32a.py > d32a/runs/make_df_aff.log 2>&1
cd $R/d32a && mkdir -p blis_bfaff
for c in 0 1; do python blis_32a.py --work ../fpt_pipeline/fpt_output_32a_bf_aff/work --dr 0.25 --c $c --exclude val9 --out blis_bfaff/c${c}_ho.npz > blis_bfaff/c${c}_ho.log 2>&1; done
echo blisbf done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32abfaff.json --steps train --fov-factors $c > log_32abfaff_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32abfaff_it30.json --steps train --fov-factors $c > log_32abfaff_it30_c$c.txt 2>&1; done
cd $R/epry_pipeline
python run_epry.py --config config_epry_32abfaff.json --steps train > log_32abfaff.txt 2>&1
python run_epry.py --config config_epry_32abfaff_1000.json --steps train > log_32abfaff_1000.txt 2>&1
echo bf512 done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32abfaff_1000.json --steps train --fov-factors $c > log_32abfaff_1000_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32abfaff_1000_it30.json --steps train --fov-factors $c > log_32abfaff_1000_it30_c$c.txt 2>&1; done
echo chainF done
