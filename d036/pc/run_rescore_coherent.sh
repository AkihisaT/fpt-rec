#!/bin/bash
# the original coherent Phase-C solutions (rings 1+3, dark mode) scored with the dark-gain metric (for continuity)
cd "$(dirname "$0")/../.."; export OMP_NUM_THREADS=1
python d018/compare_018.py --grid 512 --ds 036df --outroot d036/pc --unfiltered-metric --dark-gain --tag _dfgain > d036/pc/cmp_512_dfgain.out 2>&1
python d018/compare_018.py --grid 512 --ds 036df --outroot d036/pc --unfiltered-metric --dark-gain --tag _dfgain_ho --blis d036/blis_df13/s+0_ho.npz,d036/blis_df13/s-1_ho.npz > d036/pc/cmp_512_dfgain_ho.out 2>&1
python d018/compare_018.py --grid 1024 --ds 036df --outroot d036/pc --unfiltered-metric --dark-gain --tag _dfgain --star-centre 290,525 > d036/pc/cmp_1024_dfgain.out 2>&1
python d018/compare_018.py --grid 1024 --ds 036df --outroot d036/pc --unfiltered-metric --dark-gain --tag _dfgain_ho --star-centre 290,525 --blis d036/blis_df13/s+0_ho.npz,d036/blis_df13/s-1_ho.npz > d036/pc/cmp_1024_dfgain_ho.out 2>&1
echo "rescore done"
