#!/bin/bash
# DIP 1024 (FOV model only), mixed state and 1-mode reference, 150 iterations (+ 30-iteration runs); starts after the
# 512 DIP / EPRY partial-coherence runs have finished (waits for the last EPRY state file).
cd "$(dirname "$0")/../../dip_pipeline"
while [ ! -f ../epry_pipeline/epry_output_036pc_1024/EPRY_FOV_c1_state.npz ]; do sleep 30; done
export OMP_NUM_THREADS=2
python run_dip.py --config config_dip_036pc_1024.json --steps train --fov-factors 1 --threads 2 > run036pc_1024_c1.out 2>&1
python run_dip.py --config config_dip_036pc1m_1024.json --steps train --fov-factors 1 --threads 2 > run036pc1m_1024_c1.out 2>&1
python run_dip.py --config config_dip_036pc_1024_it30.json --steps train --fov-factors 1 --threads 2 > run036pc_1024_it30.out 2>&1
python run_dip.py --config config_dip_036pc1m_1024_it30.json --steps train --fov-factors 1 --threads 2 > run036pc1m_1024_it30.out 2>&1
echo "dip 1024 pc done"
