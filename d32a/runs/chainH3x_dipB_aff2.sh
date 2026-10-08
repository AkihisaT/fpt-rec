#!/bin/bash
export OMP_NUM_THREADS=2
cd "$(dirname "$0")/../../dip_pipeline"
for ds in 32abfaff2HB 32adfaff2HB; do python run_dip.py --config config_dip_$ds.json --steps train --fov-factors 1 > log_$ds.txt 2>&1; done
for ds in 32abfaff2HB 32adfaff2HB; do python run_dip.py --config config_dip_${ds}_it30.json --steps train --fov-factors 1 > log_${ds}_it30.txt 2>&1; done
cd ../epry_pipeline
for ds in 32abfaff2HA 32abfaff2HB 32adfaff2HA 32adfaff2HB; do python run_epry.py --config config_epry_$ds.json --steps train --fov-factors 1 > log_$ds.txt 2>&1; done
echo "chainH3x done"
