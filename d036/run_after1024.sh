#!/bin/bash
cd "$(dirname "$0")/.."
while ! grep -q "cmp df1024 done" d036/run_cmp_df1024.log 2>/dev/null; do sleep 20; done
export OMP_NUM_THREADS=4
(cd dip_pipeline && python run_dip.py --config config_dip_036df.json --steps report --threads 4 > run036df_report.out 2>&1)
(cd dip_pipeline && python run_dip.py --config config_dip_036df_1024.json --steps report --threads 4 > run036df_1024_report.out 2>&1)
python d036/tools/export_blis_df.py > d036/export_blis_df.out 2>&1
echo "after1024 done"
