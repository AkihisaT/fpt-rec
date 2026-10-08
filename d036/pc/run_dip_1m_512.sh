#!/bin/bash
# DIP 512, 1-mode reference (coherent through the partial-coherence code path), FOV then plane wave, + 30-iteration runs
cd "$(dirname "$0")/../../dip_pipeline"
export OMP_NUM_THREADS=2
python run_dip.py --config config_dip_036pc1m.json --steps train --fov-factors 1 --threads 2 > run036pc1m_c1.out 2>&1
python run_dip.py --config config_dip_036pc1m.json --steps train --fov-factors 0 --threads 2 > run036pc1m_c0.out 2>&1
python run_dip.py --config config_dip_036pc1m_it30.json --steps train --threads 2 > run036pc1m_it30.out 2>&1
echo "dip 1m 512 done"
