#!/bin/bash
# common comparison (dark-gain metric, unfiltered objects) for the partial-coherence sets; GRID=512 or 1024
cd "$(dirname "$0")/../.."; export OMP_NUM_THREADS=1
G=${GRID:-512}; SC=""; [ $G = 1024 ] && SC="--star-centre 290,525"
B=d036/pc/blis
python d018/compare_018.py --grid $G --ds 036pc --outroot d036/pc --unfiltered-metric --tag _pc $SC --blis $B/r13/s+0_pc0.1F.npz,$B/r13/s-1_pc0.1F.npz > d036/pc/cmp_${G}_pc.out 2>&1
python d018/compare_018.py --grid $G --ds 036pc --outroot d036/pc --unfiltered-metric --tag _pc_ho $SC --blis $B/r13/s+0_ho_pc0.1F.npz,$B/s-1_ho_pc0.1F.npz > d036/pc/cmp_${G}_pc_ho.out 2>&1
python d018/compare_018.py --grid $G --ds 036pc1m --outroot d036/pc --unfiltered-metric --tag _pc1m $SC --blis $B/r13/s+0_pc1mode.npz,$B/r13/s-1_pc1mode.npz > d036/pc/cmp_${G}_pc1m.out 2>&1
python d018/compare_018.py --grid $G --ds 036pc1m --outroot d036/pc --unfiltered-metric --tag _pc1m_ho $SC --blis $B/r13/s+0_ho_pc1mode.npz,$B/s-1_ho_pc1mode.npz > d036/pc/cmp_${G}_pc1m_ho.out 2>&1
echo "cmp $G done"
