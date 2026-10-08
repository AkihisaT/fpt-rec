#!/bin/bash
# 32a aff2: posaffine with the focus loop (object defocus moved into the pupil). BF set (rings 0-2). Run from the workspace root after the posaffine step has finished.
export OMP_NUM_THREADS=4
R=$PWD
python d32a/make_df_aff_32a.py fpt_output_32a_bf_aff2 work_df_aff2 > d32a/runs/make_df_aff2.log 2>&1
cd $R/d32a && mkdir -p blis_bfaff2
for c in 0 1; do python blis_32a.py --work ../fpt_pipeline/fpt_output_32a_bf_aff2/work --dr 0.25 --c $c --exclude val9 --out blis_bfaff2/c${c}_ho.npz > blis_bfaff2/c${c}_ho.log 2>&1; done
echo blisbf done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32abfaff2.json --steps train --fov-factors $c > log_32abfaff2_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32abfaff2_it30.json --steps train --fov-factors $c > log_32abfaff2_it30_c$c.txt 2>&1; done
cd $R/epry_pipeline
python run_epry.py --config config_epry_32abfaff2.json --steps train > log_32abfaff2.txt 2>&1
python run_epry.py --config config_epry_32abfaff2_1000.json --steps train > log_32abfaff2_1000.txt 2>&1
echo bf512 done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32abfaff2_1000.json --steps train --fov-factors $c > log_32abfaff2_1000_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32abfaff2_1000_it30.json --steps train --fov-factors $c > log_32abfaff2_1000_it30_c$c.txt 2>&1; done
echo chainF2 done
cd $R/fpt_pipeline && python run_pipeline.py --config config_32a_bf_aff2.json --steps linear,nonlinear,export,figures > run_bf_aff2_rest.log 2>&1; echo fpt rest done
