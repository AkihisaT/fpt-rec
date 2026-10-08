#!/bin/bash
cd "$(dirname "$0")/.."
while ! grep -q "DIP 1024 df done" dip_pipeline/run036df_1024.log 2>/dev/null; do sleep 20; done
export OMP_NUM_THREADS=4
python d018/compare_018.py --grid 1024 --ds 036df --outroot d036 --unfiltered-metric --tag _df --star-centre 290,525 > d036/compare_1024_df.out 2>&1
python d018/compare_018.py --grid 1024 --ds 036df --outroot d036 --unfiltered-metric --tag _df_ho --star-centre 290,525 --blis d036/blis_df13/s+0_ho.npz,d036/blis_df13/s-1_ho.npz > d036/compare_1024_df_ho.out 2>&1
echo "cmp df1024 done"
