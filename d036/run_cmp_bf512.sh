#!/bin/bash
export OMP_NUM_THREADS=2
python d018/compare_018.py --grid 512 --ds 036bf --outroot d036 --blis d036/objband/s+0_q3.5.npz,d036/objband/s-1_q3.5.npz --tag _objband > d036/compare_512_objband.out 2>&1
python d018/compare_018.py --grid 512 --ds 036bf --outroot d036 --blis d036/objband/s+0_q3.5_ho.npz,d036/objband/s-1_q3.5_ho.npz --tag _objband_ho > d036/compare_512_objband_ho.out 2>&1
echo cmp done
