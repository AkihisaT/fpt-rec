# Assemble the 32a deliverable folder: d32a/bundle/FPT32a_成果物まとめ_20260927/ (then copied to Drive by the caller).
# Main results = posaffine + focus loop (keys 32abfaff2 / 32adfaff2, 2026-09-28) when present, else posaffine (32abfaff / 32adfaff);
# the results before the focus loop go to before_focusloop/, the results before posaffine to before_posaffine/.
import os, shutil, glob, json
AFF2 = all(os.path.exists(f"./d32a/out_32a{s_}aff2_{g_}/compare_summary.json") for s_ in ("bf", "df") for g_ in (512, 1000))
W = "./"; B = W + "d32a/bundle/FPT32a_成果物まとめ_20260927/"
if os.path.exists(B): shutil.rmtree(B)
for d in ("figures", "tiff", "json", "scripts/d32a", "scripts/runs", "scripts/configs", "scripts/scan_fix", "pipeline_BLIS-FPM", "pipeline_changes/fptrecon",
          "supplement_astigmatism", "before_posaffine/figures", "before_posaffine/tiff", "before_posaffine/json") + (("before_focusloop/figures", "before_focusloop/tiff", "before_focusloop/json", "supplement_focus_in_pupil/scripts") if AFF2 else ()):
    os.makedirs(B + d, exist_ok=True)
cp = lambda a, b: shutil.copy2(W + a, B + b)
ex = lambda a: os.path.exists(W + a)
cp("d32a/32a_report.md", "32a_report.md"); cp("d32a/deck/FPT32a_再構成まとめ.pptx", "FPT32a_再構成まとめ.pptx")
for f in ("fig_geometry_32a.png", "fig_fov_dr_32a.png", "fig_astig_origin_32a.png"): cp("d32a/" + f, "figures/" + f)
PA1 = "fpt_pipeline/fpt_output_32a_bf_aff/"
PA = "fpt_pipeline/fpt_output_32a_bf_aff2/" if AFF2 else PA1
cp(PA + "figures/fig0_position_affine.png", "figures/fig0_position_affine.png"); cp(PA + "results/position_affine.json", "json/position_affine.json")
def put(conds, fig_d, tif_d, js_d):
    done = []
    for k, pre in conds.items():
        o = W + f"d32a/out_{k}/"
        if not os.path.exists(o + "compare_summary.json"): continue
        done.append(pre)
        for f in ("images_FOV_c1", "images_planewave", "zoom", "spokes", "misfit"):
            if os.path.exists(o + f"fig_{f}.png"): shutil.copy2(o + f"fig_{f}.png", B + f"{fig_d}/{pre}_{f}.png")
        shutil.copytree(o + "tiff", B + f"{tif_d}/{pre}")
        shutil.copy2(o + "compare_summary.json", B + f"{js_d}/{pre}_compare_summary.json"); shutil.copy2(o + "spokes.json", B + f"{js_d}/{pre}_spokes.json")
    return done
T_ = "aff2" if AFF2 else "aff"
done = put({f"32abf{T_}_512": "bf512", f"32abf{T_}_1000": "bf1000", f"32adf{T_}_512": "df512", f"32adf{T_}_1000": "df1000"}, "figures", "tiff", "json")
if AFF2:
    put({"32abfaff_512": "bf512", "32abfaff_1000": "bf1000", "32adfaff_512": "df512", "32adfaff_1000": "df1000"}, "before_focusloop/figures", "before_focusloop/tiff", "before_focusloop/json")
    cp(PA1 + "figures/fig0_position_affine.png", "before_focusloop/figures/fig0_position_affine.png"); cp(PA1 + "results/position_affine.json", "before_focusloop/json/position_affine.json")
    if ex("d32a/32a_report_v2_aff.md"): cp("d32a/32a_report_v2_aff.md", "before_focusloop/32a_report_before_focusloop.md")
    for f in glob.glob(W + "d32a/blis_*aff/c*_ho*.json"): shutil.copy2(f, B + "before_focusloop/json/" + "blis_" + f.split("/")[-2].replace("blis_", "") + "_" + os.path.basename(f))
    if ex("d32a/work_df_aff/calibration.json"): cp("d32a/work_df_aff/calibration.json", "before_focusloop/json/work_df_aff__calibration.json")
    fo = W + PA1; dst = "before_focusloop/pipeline_BLIS-FPM"
    shutil.copytree(fo + "results", B + dst + "/results", dirs_exist_ok=True); shutil.copytree(fo + "figures", B + dst + "/figures", dirs_exist_ok=True)
    for f in ("calibration.json",): shutil.copy2(fo + "work/" + f, B + dst + "/" + f)
    if os.path.exists(fo + "pipeline.log"): shutil.copy2(fo + "pipeline.log", B + dst + "/pipeline.log")
    for f in ("fig_focusloop_32a.png", "fig_focusloop_zoom_32a_512.png", "fig_focusloop_field_32a_512.png", "fig_focusloop_zoom_32a_1000.png", "fig_focusloop_field_32a_1000.png", "frc/fig_mtf_signed_aff_vs_aff2_512.png", "frc/fig_mtf_signed_aff_vs_aff2_1000.png", "frc/fig_frc_aff_vs_aff2_512.png"):
        if ex("d32a/" + f): cp("d32a/" + f, "figures/" + os.path.basename(f)); cp("d32a/" + f, "supplement_focus_in_pupil/" + os.path.basename(f))
    for f in ("focusloop_summary_32a.json", "fig_focusloop_images_32a_512.json", "fig_focusloop_images_32a_1000.json", "frc/frc_mtf_32a_aff2.json", "frc/resolution_table_32a_aff2.json", "af_test/pupil_basis_test.json", "af_test/pupil_annulus_quadratic.json",
              "frc/README_focus_in_pupil_32a.md"):
        if ex("d32a/" + f): cp("d32a/" + f, "supplement_focus_in_pupil/" + os.path.basename(f))
    cp(PA + "results/position_affine.json", "supplement_focus_in_pupil/position_affine_aff2.json"); cp(PA1 + "results/position_affine.json", "supplement_focus_in_pupil/position_affine_aff.json")
    for f in ("af_test/pupil_basis_test.py", "af_test/pupil_annulus_quadratic.py", "fig_focusloop_32a.py", "fig_focusloop_images_32a.py", "frc/frc_mtf_32a.py", "frc/fig_resolution_aff2.py", "frc/setup_halves.py", "frc/make_readme_focus_in_pupil.py",
              "make_df_aff_32a.py", "dip_df_shift_stats.py", "runs/chainF2_bfaff2.sh", "runs/chainG2_dfaff2.sh", "runs/post_aff2.sh", "runs/chainH1x_blis_aff2.sh", "runs/chainH2x_dipA_aff2.sh", "runs/chainH3x_dipB_aff2.sh"):
        if ex("d32a/" + f): cp("d32a/" + f, "supplement_focus_in_pupil/scripts/" + os.path.basename(f))
old = put({"32abf_512": "bf512", "32abf_1000": "bf1000", "32adf_512": "df512", "32adf_1000": "df1000"}, "before_posaffine/figures", "before_posaffine/tiff", "before_posaffine/json")
if ex("d32a/32a_report_before_posaffine.md"): cp("d32a/32a_report_before_posaffine.md", "before_posaffine/32a_report_before_posaffine.md")
for f in ("geom_direct_v2.json", "register_32a.json", "dark_estimate.json", "fov_dr_summary.json", "df_registration_test.json", "32a_summary.json", "dr_cmp/dr_compare_summary.json",
          "drift_linear_fit.json", "aniso_compare_32a.json", ("work_df_aff2/calibration.json" if AFF2 else "work_df_aff/calibration.json"), "dip_df_shift_stats.json", "focusloop_summary_32a.json"):
    if ex("d32a/" + f): shutil.copy2(W + "d32a/" + f, B + "json/" + f.replace("/", "__"))
for f in glob.glob(W + ("d32a/blis_*aff2/c*_ho*.json" if AFF2 else "d32a/blis_*aff/c*_ho*.json")): shutil.copy2(f, B + "json/" + "blis_" + f.split("/")[-2].replace("blis_", "") + "_" + os.path.basename(f))
for f in glob.glob(W + "d32a/blis_bf/c*_ho*.json") + glob.glob(W + "d32a/blis_df/c*_ho*.json"):
    shutil.copy2(f, B + "before_posaffine/json/" + "blis_" + f.split("/")[-2].replace("blis_", "") + "_" + os.path.basename(f))
# pipeline (BLIS-FPM, all 60 images) after posaffine; the former run under before_posaffine/
for fo, dst in ((W + PA, "pipeline_BLIS-FPM"), (W + "fpt_pipeline/fpt_output_32a_bf_reg2/", "before_posaffine/pipeline_BLIS-FPM")):
    shutil.copytree(fo + "results", B + dst + "/results", dirs_exist_ok=True); shutil.copytree(fo + "figures", B + dst + "/figures", dirs_exist_ok=True)
    for f in ("calibration.json", "register_to_model.json"):
        if os.path.exists(fo + "work/" + f): shutil.copy2(fo + "work/" + f, B + dst + "/" + f)
    if os.path.exists(fo + "pipeline.log"): shutil.copy2(fo + "pipeline.log", B + dst + "/pipeline.log")
# pipeline changes (the same files were added to the user's pipeline folder, originals kept as *.bak_20260927)
for f in ("fptrecon/posaffine.py", "fptrecon/nonlinear.py", "run_pipeline.py", "config_32a_bf_aff.json", "config_32a_bf_aff2.json", "README.md"): cp("fpt_pipeline/" + f, "pipeline_changes/" + f)
# astigmatism supplement
for f in ("aniso_32a.json", "aniso_32a.py", "aniso_compare_32a.py", "kept_linear_field.json", "fig_astig_origin_32a.png", "astig_test/astig_test_summary.json", "astig_test/autofocus_keep_nolin.json", "af_test/parallax_test.py", "af_test/parallax_test.json", "af_test/af_bias_test.py", "af_test/af_bias_test.json",
          "astig_test/psd_axis_bands.json", "astig_test/psd_directional.json", "astig_test/psd_directional_blis_test.json"):
    if ex("d32a/" + f): cp("d32a/" + f, "supplement_astigmatism/" + os.path.basename(f))
for f in glob.glob(W + "d32a/*.py"): shutil.copy2(f, B + "scripts/d32a/")
for f in glob.glob(W + "d32a/scan_fix/*.*"): shutil.copy2(f, B + "scripts/scan_fix/")
for f in glob.glob(W + "d32a/runs/*.sh"): shutil.copy2(f, B + "scripts/runs/")
for f in glob.glob(W + "*/config_*32a*.json"): shutil.copy2(f, B + "scripts/configs/" + f.split("/")[-2] + "__" + os.path.basename(f))
for f in ("work_df_nd/calibration.json", "work_df_nd/prep_df.json", "work_df_nd/noise.json", "data_reg/positions_um.csv", "data_reg/prep_reg.json"):
    if ex("d32a/" + f): shutil.copy2(W + "d32a/" + f, B + "json/" + f.replace("/", "__"))
n = sum(len(fs) for _, _, fs in os.walk(B)); sz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(B) for f in fs)
print("conditions", done, "before", old, "files", n, f"{sz / 1e6:.0f} MB")
