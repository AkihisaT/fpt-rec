#!/bin/bash
# 32a bright field (rings 0-2, registered, pupil calibration with astigmatism): BLIS-FPM (pipeline, all images) and
# BLIS-FPM with the 7 held-out images removed, DIP 512 (600 it, 30 it), EPRY 512 / 1000. Run from the workspace root.
export OMP_NUM_THREADS=4
R=$PWD
cd $R/fpt_pipeline && python run_pipeline.py --config config_32a_bf_reg2.json --steps linear,nonlinear,export,figures > run_bf_reg2_v2.log 2>&1
cd $R/d32a && mkdir -p blis_bf && for c in 0 1; do python blis_32a.py --work ../fpt_pipeline/fpt_output_32a_bf_reg2/work --dr 0.25 --c $c --exclude val9 --out blis_bf/c${c}_ho.npz > blis_bf/c${c}_ho.log 2>&1; done
cd $R/dip_pipeline
for c in 0 1; do python run_dip.py --config config_dip_32abf.json --steps train --fov-factors $c > log_32abf_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32abf_it30.json --steps train --fov-factors $c > log_32abf_it30_c$c.txt 2>&1; done
cd $R/epry_pipeline
python run_epry.py --config config_epry_32abf.json --steps train > log_32abf.txt 2>&1
python run_epry.py --config config_epry_32abf_1000.json --steps train > log_32abf_1000.txt 2>&1
echo chainC done
