#!/bin/bash
# 036 partial coherence: choose the halo fraction eta on held-out images (BLIS-FPM, rings 1+3, FOV c=1, 4 images held out)
cd "$(dirname "$0")/.."
export BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$PWD/pc/blis
for pc in 1mode 0.1 0.3; do
  T=3; [ $pc = 1mode ] && T=2; OMP_NUM_THREADS=$T BLIS_DF_PC=$pc python blis_df.py -1 $T 0,25,34,43 > pc/blis_sel_$pc.out 2>&1 &
done
wait
echo "select eta done"
