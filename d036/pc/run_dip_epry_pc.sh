#!/bin/bash
# DIP 512 and EPRY 512/1024 with the chosen eta (configs 036pc from make_pc_configs.py), 2 threads
cd "$(dirname "$0")/../../dip_pipeline"; export OMP_NUM_THREADS=2
python run_dip.py --config config_dip_036pc.json --steps train --fov-factors 1 --threads 2 > run036pc_c1.out 2>&1
python run_dip.py --config config_dip_036pc.json --steps train --fov-factors 0 --threads 2 > run036pc_c0.out 2>&1
python run_dip.py --config config_dip_036pc_it30.json --steps train --threads 2 > run036pc_it30.out 2>&1
echo "dip pc 512 done"
cd ../epry_pipeline
python run_epry.py --config config_epry_036pc.json --steps train --threads 2 > run036pc_epry.out 2>&1
python run_epry.py --config config_epry_036pc_1024.json --steps train --threads 2 > run036pc_epry_1024.out 2>&1
echo "epry pc done"
