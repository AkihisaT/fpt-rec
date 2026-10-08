#!/bin/bash
W=$(cd "$(dirname "$0")/.." && pwd); cd $W/dip_pipeline
while ! grep -q "1024 done" run036bf_1024.log 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=3
python run_dip.py --config config_dip_036df.json --steps train --fov-factors 0 --threads 3 > run036df_c0.out 2>&1 &
python run_dip.py --config config_dip_036df.json --steps train --fov-factors 1 --threads 3 > run036df_c1.out 2>&1
wait
python run_dip.py --config config_dip_036df_it30.json --steps train --fov-factors 0 --threads 3 > run036df_it30_c0.out 2>&1 &
python run_dip.py --config config_dip_036df_it30.json --steps train --fov-factors 1 --threads 3 > run036df_it30_c1.out 2>&1
wait
echo "DIP phase C done"
