#!/bin/bash
# 32a bright field: DIP 30 it (512), EPRY (512, 1000), DIP 1000 plane wave. Run from the workspace root.
set -e
export OMP_NUM_THREADS=4
cd dip_pipeline
until tail -1 log_32abf_c1.txt | grep -q "done"; do sleep 20; done
python run_dip.py --config config_dip_32abf_it30.json --steps train --fov-factors 0 > log_32abf_it30_c0.txt 2>&1
python run_dip.py --config config_dip_32abf_it30.json --steps train --fov-factors 1 > log_32abf_it30_c1.txt 2>&1
cd ../epry_pipeline
python run_epry.py --config config_epry_32abf.json --steps train > log_32abf.txt 2>&1
python run_epry.py --config config_epry_32abf_1000.json --steps train > log_32abf_1000.txt 2>&1
cd ../dip_pipeline
python run_dip.py --config config_dip_32abf_1000.json --steps train --fov-factors 0 > log_32abf_1000_c0.txt 2>&1
echo chainA done
