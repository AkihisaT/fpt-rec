#!/bin/bash
# aff2 (focus loop): split-half runs, comparisons of the 32 full reconstructions, split halves, signed MTF / FRC.
# Starts after chainF2 and chainG2. Run from the workspace root.
R=$PWD
until grep -q "chainF2 done" d32a/runs/chainF2_bfaff2.log 2>/dev/null && grep -q "chainG2 done" d32a/runs/chainG2_dfaff2.log 2>/dev/null; do sleep 60; done
echo "start $(date +%H:%M)"
bash d32a/runs/chainH1x_blis_aff2.sh > d32a/runs/chainH1x.log 2>&1 &
bash d32a/runs/chainH2x_dipA_aff2.sh > d32a/runs/chainH2x.log 2>&1 &
bash d32a/runs/chainH3x_dipB_aff2.sh > d32a/runs/chainH3x.log 2>&1 &
export OMP_NUM_THREADS=2
for g in 512 1000; do
  if [ $g = 512 ]; then L="centre 512 px"; else L="full field 1000 px"; fi
  python d32a/compare_32a.py --ds 32abfaff2 --grid $g --blis d32a/blis_bfaff2/c0_ho.npz,d32a/blis_bfaff2/c1_ho.npz > d32a/runs/compare_bfaff2$g.log 2>&1
  python d32a/fig_32a.py d32a/out_32abfaff2_$g "BF (rings 0-2), drift-corrected, focus in pupil, $L — " >> d32a/runs/compare_bfaff2$g.log 2>&1
  python d32a/compare_32a.py --ds 32adfaff2 --grid $g --blis d32a/blis_dfaff2/c0_ho.npz,d32a/blis_dfaff2/c1_ho.npz > d32a/runs/compare_dfaff2$g.log 2>&1
  python d32a/fig_32a.py d32a/out_32adfaff2_$g "BF+DF (rings 0-2 + 6-7), drift-corrected, focus in pupil, $L — " >> d32a/runs/compare_dfaff2$g.log 2>&1
  echo "compare $g done $(date +%H:%M)"
done
wait
for h in A B; do for s in bf df; do
  python d32a/compare_32a.py --ds 32a${s}aff2H$h --grid 512 --blis d32a/blis_halves2/none.npz,d32a/blis_halves2/${s}${h}_c1.npz > d32a/runs/compare_${s}aff2H$h.log 2>&1
done; done
python d32a/frc/frc_mtf_32a.py aff2 > d32a/runs/frc_mtf_aff2.log 2>&1
echo "post done $(date +%H:%M)"
