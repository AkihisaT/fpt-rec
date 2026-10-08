#!/bin/bash
# BLIS-FPM extras for 036 bright field: FOV-strength scan (70 it) and object band-limit runs (all images / 2 held out)
export FPT_CFG=$PWD/fpt_pipeline/config_036bf.json FPT_WORK=$PWD/fpt_pipeline/fpt_output_036bf/work
export OMP_NUM_THREADS=2
FPT_OUT=$PWD/d036/kappa_scan python d018/tools/kappa_scan.py 0,-0.5,-1,-1.5,-2,-2.5,1 2
for s in 0 -1; do
  FPT_OUT=$PWD/d036/objband python d018/tools/blis_objband.py $s 3.5 2
  FPT_OUT=$PWD/d036/objband python d018/tools/blis_objband.py $s 3.5 2 0,5
done
echo extras done
