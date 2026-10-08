#!/bin/bash
# 32a main workflow (parallel-beam illumination + objective-FZP scan, 30 keV, dr 250 nm), package layout.
# This is the order in which the results of the 32a report (aff2 = mechanical drift correction + focus loop) were made,
# written as one script (the chain*.sh files it calls are the ones used in the analysis; their wait loops look for the
# "... done" lines of the logs written here).
# Run from the package root:
#   export FPT32A_RAW=/path/to/32a
#   bash d32a/runs/run_32a_main.sh > d32a/runs/run_32a_main.log 2>&1
# Environments: fpt_pipeline/environment.yml (steps 1-5, BLIS-FPM) and dip_pipeline/environment_dip.yml (DIP, EPRY);
# the analysis ran everything in the dip environment (it contains the packages of both).
# Run time in the analysis (8 CPU cores): chainF2 + chainG2 in parallel 2.6 h, post_aff2.sh 20 min (DIP 1000 px ~6.5 s/iteration).
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
echo "start $(date)"
# 1. dark offset + lens/detector half-pulse registration -> d32a/data_reg; BF subset (rings 0-2, 60 images) -> d32a/data_reg_bf
python d32a/prep_32a_reg.py
python d32a/make_subset_32a.py data_reg_bf 0:60
# 2. illumination geometry from the vignetting of the direct frames -> d32a/geom_direct_v2.json (A, c0, R_field)
python d32a/fit_geom_direct_32a.py
# 3. fpt_pipeline preprocessing of the BF set; calibration (k_n, kappa) from the geometry, dr 0.25 um, initial defocus 5 mm
(cd fpt_pipeline && python run_pipeline.py --config config_32a_bf.json --steps preprocess)
python d32a/make_calib_32a.py fpt_pipeline/fpt_output_32a_bf/work 0.25 5000 0:60
# 4. posaffine: mechanical drift (linear in the lens position) + random shifts, contrast focus, focus loop
mkdir -p fpt_pipeline/fpt_output_32a_bf_aff2/work
cp fpt_pipeline/fpt_output_32a_bf/work/preprocessed.npy fpt_pipeline/fpt_output_32a_bf/work/preprocess_meta.npz \
   fpt_pipeline/fpt_output_32a_bf/work/calibration.json fpt_pipeline/fpt_output_32a_bf_aff2/work/
(cd fpt_pipeline && python run_pipeline.py --config config_32a_bf_aff2.json --steps posaffine > run_bf_aff2_posaffine.log 2>&1)
# 5. dark-field data (rings 6-7, 95 images) -> d32a/work_df (make_df_aff_32a.py in chainF2 shifts it with the drift model)
python d32a/prep_32a_df.py 0.25 5000
# 6. BLIS-FPM (d32a/blis_32a.py, 1 image held out), DIP (optimum + 30 iterations), EPRY; plane wave (c 0) and FOV model (c 1);
#    512 px centre and 1000 px full field.  chainF2 = BF set (and at the end the fpt_pipeline linear/nonlinear/export/figures),
#    chainG2 = BF+DF set (d32a/work_df_aff2)
#    (run in parallel as in the analysis; chainG2 waits for the "blisbf done" line of the chainF2 log)
bash d32a/runs/chainF2_bfaff2.sh > d32a/runs/chainF2_bfaff2.log 2>&1 &
sleep 5
bash d32a/runs/chainG2_dfaff2.sh > d32a/runs/chainG2_dfaff2.log 2>&1 &
wait
# 7. split halves (FRC), comparison of the 32 reconstructions, figures, signed MTF / FRC
python d32a/frc/setup_halves.py aff2
mkdir -p d32a/blis_halves2
bash d32a/runs/post_aff2.sh > d32a/runs/post_aff2.log 2>&1
# 8. before/after figures of the focus loop need the aff (no focus loop) results as well: config_32a_bf_aff.json and
#    chainF_bfaff.sh / chainG_dfaff.sh (same steps with work_df_aff); then python d32a/fig_focusloop_32a.py and
#    python d32a/fig_focusloop_images_32a.py.  Report / slides: python d32a/make_report_32a.py; python d32a/build_slides_32a.py
echo "done $(date)"
