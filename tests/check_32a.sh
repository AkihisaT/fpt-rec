#!/bin/bash
# Check on a new computer: 32a steps 1-5 (registration, BF subset, illumination geometry, preprocessing, calibration,
# posaffine with the focus loop), then comparison with the result of the analysis (tests/reference/).
# About 15 min on an 8-core Mac. Needs the 32a raw data:
#   source paths_local.sh      (FPT32A_RAW)
#   conda activate dip
#   bash tests/check_32a.sh > tests/check_32a.log 2>&1
set -e
: "${FPT32A_RAW:?set FPT32A_RAW (see paths_local.example.sh)}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
echo "start $(date)"
python d32a/prep_32a_reg.py
python d32a/make_subset_32a.py data_reg_bf 0:60
python d32a/fit_geom_direct_32a.py --check          # compares with d32a/geom_direct_v2.json (does not overwrite it)
(cd fpt_pipeline && python run_pipeline.py --config config_32a_bf.json --steps preprocess)
python d32a/make_calib_32a.py fpt_pipeline/fpt_output_32a_bf/work 0.25 5000 0:60
W=fpt_pipeline/fpt_output_32a_bf_aff2/work
mkdir -p $W && rm -f $W/preprocessed_before_posaffine.npy
cp fpt_pipeline/fpt_output_32a_bf/work/preprocessed.npy fpt_pipeline/fpt_output_32a_bf/work/preprocess_meta.npz \
   fpt_pipeline/fpt_output_32a_bf/work/calibration.json $W/
(cd fpt_pipeline && python run_pipeline.py --config config_32a_bf_aff2.json --steps posaffine)
grep "focus loop" fpt_pipeline/fpt_output_32a_bf_aff2/pipeline.log | tail -3 || true
python tests/compare_ref.py
echo "done $(date)"
