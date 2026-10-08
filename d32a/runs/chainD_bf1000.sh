#!/bin/bash
# 32a bright field, full field 1000 px: DIP 300 it and 30 it (after chainB). Run from the workspace root.
export OMP_NUM_THREADS=4
until grep -q "chainB done" d32a/runs/chainB_blis.log; do sleep 30; done
cd dip_pipeline; rm -f dip_output_32abf_1000/STOP
for c in 0 1; do python run_dip.py --config config_dip_32abf_1000.json --steps train --fov-factors $c > log_32abf_1000_c$c.txt 2>&1; done
for c in 0 1; do python run_dip.py --config config_dip_32abf_1000_it30.json --steps train --fov-factors $c > log_32abf_1000_it30_c$c.txt 2>&1; done
echo chainD done
