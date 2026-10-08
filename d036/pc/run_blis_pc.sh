#!/bin/bash
# 036 partial coherence: BLIS-FPM runs after the choice of eta (ETA=<eta> bash run_blis_pc.sh); two jobs at a time, 3 threads each.
# Mixed state (BLIS_DF_PC=$ETA) and 1-mode reference (BLIS_DF_PC=1mode) with the same gain handling.
cd "$(dirname "$0")/.."
OUT=$PWD/pc/blis; export OMP_NUM_THREADS=3
job() {  # rings pc modes s excl tag
  BLIS_DF_RINGS=$1 BLIS_DF_PC=$2 BLIS_DF_PCMODES=$3 BLIS_DF_OUT=$OUT/$6 python blis_df.py $4 3 $5 > pc/blis_$6_$2_s$4_$( [ "$5" = "-" ] && echo full || echo ho ).out 2>&1
}
mkdir -p $OUT/r13 $OUT/r123 $OUT/r1
# 28 images (rings 1 + 3): FOV and plane wave, full solution; plane-wave hold-out (the FOV hold-out runs are the eta selection)
job 1,3 $ETA F -1 - r13 & job 1,3 1mode F -1 - r13 & wait
job 1,3 $ETA F 0 - r13 & job 1,3 1mode F 0 - r13 & wait
job 1,3 $ETA F 0 0,25,34,43 r13 & job 1,3 1mode F 0 0,25,34,43 r13 & wait
echo "r13 done"
# 44 images (rings 1-3, ring 2 included): FOV, finer mode set A; hold-out = every 9th image
job 1,2,3 $ETA A -1 0,9,18,27,36 r123 & job 1,2,3 1mode A -1 0,9,18,27,36 r123 & wait
job 1,2,3 $ETA A -1 - r123 & job 1,2,3 1mode A -1 - r123 & wait
echo "r123 done"
# 9 bright-field images (ring 1): FOV and plane wave, full and hold-out (images 0, 5)
for s in -1 0; do
  job 1 $ETA F $s - r1 & job 1 1mode F $s - r1 & wait
  job 1 $ETA F $s 0,5 r1 & job 1 1mode F $s 0,5 r1 & wait
done
echo "blis pc all done"
