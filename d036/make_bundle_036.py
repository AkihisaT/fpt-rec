# -*- coding: utf-8 -*-
# Assemble the FPT #036 deliverable folder (Drive copy):  python d036/make_bundle_036.py -> d036/bundle/FPT036_成果物まとめ_20260926/
import glob, json, os, re, shutil, sys, zipfile
W = "./"
NAME = "FPT036_成果物まとめ_20260926"
B = W + f"d036/bundle/{NAME}/"
FIGS = {  # figures/<name> : workspace source (report links are rewritten to these names)
    "036_fig_data.png": "d036/036_fig_data.png", "036_fig_darkfield.png": "d036/036_fig_darkfield.png",
    "036_fig_fov_scan.png": "d036/036_fig_fov_scan.png", "036_fig_spokes.png": "d036/036_fig_spokes.png", "036_fig_df_starzoom.png": "d036/036_fig_df_starzoom.png",
    "BLIS_bf_fig1_calibration.png": "fpt_pipeline/fpt_output_036bf/figures/fig1_calibration.png",
    "BLIS_bf_fig2_reconstruction.png": "fpt_pipeline/fpt_output_036bf/figures/fig2_reconstruction.png",
    "BLIS_bf_fig3_pupil.png": "fpt_pipeline/fpt_output_036bf/figures/fig3_pupil.png",
    "BLIS_bf_fig4_fov_effect.png": "fpt_pipeline/fpt_output_036bf/figures/fig4_fov_effect.png",
    "bf_512_compare_FOV.png": "d036/out_512/fig_compare_FOV_c1.png", "bf_512_compare_planewave.png": "d036/out_512/fig_compare_planewave.png",
    "bf_1024_compare_FOV.png": "d036/out_1024/fig_compare_FOV_c1.png", "bf_1024_compare_planewave.png": "d036/out_1024/fig_compare_planewave.png",
    "df_512_compare_FOV.png": "d036/out_512_df/fig_compare_FOV_c1.png", "df_512_compare_planewave.png": "d036/out_512_df/fig_compare_planewave.png",
    "df_1024_compare_FOV.png": "d036/out_1024_df/fig_compare_FOV_c1.png", "df_1024_compare_planewave.png": "d036/out_1024_df/fig_compare_planewave.png",
    "bf_512_dip_training.png": "dip_pipeline/dip_output_036bf/fig_dip_training.png",
    "bf_1024_dip_training.png": "dip_pipeline/dip_output_036bf_1024/fig_dip_training.png",
    "df_512_dip_training.png": "dip_pipeline/dip_output_036df/fig_dip_training.png",
    "df_1024_dip_training.png": "dip_pipeline/dip_output_036df_1024/fig_dip_training.png",
    "bf_512_epry_sweeps.png": "epry_pipeline/epry_output_036bf/fig_epry_sweeps.png",
    "bf_1024_epry_sweeps.png": "epry_pipeline/epry_output_036bf_1024/fig_epry_sweeps.png",
    "036_fig_pc_direct.png": "d036/036_fig_pc_direct.png", "036_fig_pc_compare.png": "d036/036_fig_pc_compare.png",
    "036_fig_pc_starzoom.png": "d036/036_fig_pc_starzoom.png",
    "pc_512_compare_FOV.png": "d036/pc/out_512_pc/fig_compare_FOV_c1.png", "pc1mode_512_compare_FOV.png": "d036/pc/out_512_pc1m/fig_compare_FOV_c1.png",
    "pc_1024_compare_FOV.png": "d036/pc/out_1024_pc/fig_compare_FOV_c1.png", "pc1mode_1024_compare_FOV.png": "d036/pc/out_1024_pc1m/fig_compare_FOV_c1.png",
}
LINKS = {"036_fig_data.png": "036_fig_data.png", "036_fig_darkfield.png": "036_fig_darkfield.png",
         "out_512/fig_compare_FOV_c1.png": "bf_512_compare_FOV.png", "out_1024/fig_compare_FOV_c1.png": "bf_1024_compare_FOV.png",
         "out_512/fig_compare_planewave.png": "bf_512_compare_planewave.png", "out_1024/fig_compare_planewave.png": "bf_1024_compare_planewave.png"}
TIF = lambda d, pre="": sorted(p for p in glob.glob(W + d + "/*.tif") if os.path.basename(p).startswith(pre))
J = lambda p, new=None: (W + p, new)
COPY = {
    "01_データと前処理": [J("d036/data/prep_036.json"), J("d036/work_df/prep_df.json"), J("d036/work_df/noise.json"),
                        J("d036/data/illumination_036.json"), J("d036/data/illum_level.json"), J("d036/data/kappa_direct_fit.json"),
                        J("fpt_pipeline/fpt_output_036bf/work/calibration.json", "calibration_ring1.json"),
                        J("d036/work_df/calibration.json", "calibration_44images.json")],
    "02_明視野_512": [J("d036/out_512/compare_summary.json"), J("d036/out_512_objband/compare_summary.json", "compare_summary_blis_objband.json"),
                    J("d036/out_512_objband_ho/compare_summary.json", "compare_summary_blis_objband_holdout.json")]
                   + [(p, None) for p in TIF("dip_pipeline/dip_output_036bf")] + [J("dip_pipeline/dip_output_036bf/dip_summary.json")]
                   + [(p, os.path.basename(p).replace("DIP30_", "DIP30it_")) for p in TIF("d036/out_512/tiff", "DIP30_")]
                   + [(p, None) for p in TIF("epry_pipeline/epry_output_036bf")] + [J("epry_pipeline/epry_output_036bf/epry_summary.json")],
    "03_明視野_1024": [J("d036/out_1024/compare_summary.json"), J("d036/out_1024/spokes.json")]
                    + [(p, None) for p in TIF("dip_pipeline/dip_output_036bf_1024")] + [J("dip_pipeline/dip_output_036bf_1024/dip_summary.json")]
                    + [(p, os.path.basename(p).replace("DIP30_", "DIP30it_")) for p in TIF("d036/out_1024/tiff", "DIP30_")]
                    + [(p, None) for p in TIF("epry_pipeline/epry_output_036bf_1024")] + [J("epry_pipeline/epry_output_036bf_1024/epry_summary.json")],
    "04_BLIS-FPM_明視野": [(p, None) for p in glob.glob(W + "fpt_pipeline/fpt_output_036bf/results/*")]
                        + [J("fpt_pipeline/fpt_output_036bf/work/nonlinear_cp0_00_bandcheck.json", "bandcheck_planewave.json"),
                           J("fpt_pipeline/fpt_output_036bf/work/nonlinear_cp1_00_bandcheck.json", "bandcheck_FOV_c1.json"),
                           J("d036/kappa_scan/scan.json", "fov_factor_scan_70it.json")]
                        + [(p, "objband_" + os.path.basename(p)) for p in glob.glob(W + "d036/objband/*.json")],
    "05_暗視野_512": [J("d036/out_512_df/compare_summary.json", "compare_summary_unfiltered_metric.json"),
                    J("d036/out_512_df_ho/compare_summary.json", "compare_summary_blis_holdout.json"),
                    J("d036/out_512_df_bp/compare_summary.json", "compare_summary_bandpassed_metric.json")]
                   + [(p, None) for p in TIF("dip_pipeline/dip_output_036df")]
                   + [(p, os.path.basename(p).replace("DIP30_", "DIP30it_")) for p in TIF("d036/out_512_df/tiff", "DIP30_")]
                   + [(p, None) for p in TIF("d036/out_512_df/tiff", "EPRY_")] + [(p, None) for p in TIF("d036/blis_df13/export")],
    "06_暗視野_1024": [J("d036/out_1024_df/compare_summary.json", "compare_summary_unfiltered_metric.json"),
                     J("d036/out_1024_df_ho/compare_summary.json", "compare_summary_blis_holdout.json"), J("d036/out_1024_df/spokes.json")]
                    + [(p, None) for p in TIF("dip_pipeline/dip_output_036df_1024")]
                    + [(p, os.path.basename(p).replace("DIP30_", "DIP30it_")) for p in TIF("d036/out_1024_df/tiff", "DIP30_")]
                    + [(p, None) for p in TIF("d036/out_1024_df/tiff", "EPRY_")],
    "07_暗視野_診断": [(p, "blis_44images_" + os.path.basename(p)) for p in glob.glob(W + "d036/blis_df/*.json")]
                    + [(p, "blis_rings13_" + os.path.basename(p)) for p in glob.glob(W + "d036/blis_df13/*.json")]
                    + [(p, "blis_ring3test_" + os.path.basename(p)) for p in glob.glob(W + "d036/blis_k3test/*.json")]
                    + [(p, "epry_variant_" + os.path.basename(p)) for p in glob.glob(W + "d036/epry_diag/*.json")]
                    + [J("d036/frc_halves.json")],
    "09_部分コヒーレンス": [J("d036/pc/" + f) for f in ("pc_summary.json", "direct_fit.json", "direct_fit_c1.json", "direct_fit_common.json", "S_fit.json",
                          "S_fit_c1.json", "halo_fit.json", "predict_pc_gain.json", "modes_convergence.json", "modes_adaptive_eta0.3.json", "synthetic_test.json")]
                    + [J("d036/pc/predict_pc.json", "predict_pc_nogain.json")]
                    + [(W + f"d036/pc/out_{t}/compare_summary.json", f"compare_summary_{t}.json") for t in
                       ("512_pc", "512_pc_ho", "512_pc1m", "512_pc1m_ho", "512_dfgain", "512_dfgain_ho", "1024_pc", "1024_pc_ho", "1024_pc1m", "1024_pc1m_ho", "1024_dfgain", "1024_dfgain_ho")]
                    + [(W + f"d036/pc/out_{t}/spokes.json", f"spokes_{t}.json") for t in ("1024_pc", "1024_pc1m")]
                    + [(p, "blis_" + os.path.relpath(p, W + "d036/pc/blis").replace("/", "_")) for p in sorted(glob.glob(W + "d036/pc/blis/**/*.json", recursive=True))],
    "09_部分コヒーレンス/tiff_1024_partial_coherent": [(p, None) for p in TIF("d036/pc/out_1024_pc/tiff")],
    "09_部分コヒーレンス/tiff_1024_1mode": [(p, None) for p in TIF("d036/pc/out_1024_pc1m/tiff")],
}
PIPE = {
    "fpt_pipeline.zip": ("fpt_pipeline", ["README.md", "environment.yml", "config_036bf.json", "run_pipeline.py", "fptrecon/*.py"]),
    "dip_pipeline.zip": ("dip_pipeline", ["README.md", "environment_dip.yml", "config_dip_036*.json", "run_dip.py", "dipfpm/*.py", "tools/*.py"]),
    "epry_pipeline.zip": ("epry_pipeline", ["README.md", "environment_epry.yml", "config_epry_036*.json", "run_epry.py", "eprfpm/*.py", "tools/*.py"]),
    "d036_scripts.zip": ("d036", ["*.py", "*.sh", "tools/*.py", "pc/*.py", "pc/*.sh"]),
    "d018_tools_used.zip": ("d018", ["compare_018.py", "tools/blis_objband.py", "tools/kappa_scan.py", "tools/blis_bandcheck.py"]),
}
FILES_MD = f"""
## フォルダの構成（`{NAME}/`）

| 場所 | 内容 |
|---|---|
| `036_report.md` / `FPT036_再構成まとめ.pptx` / `036_summary.json` | レポート、スライド（16:9）、全数値 |
| `figures/` | 図（`bf_*` 明視野のみ、`df_*` 暗視野を含む 28 枚、`pc_*`・`pc1mode_*` 部分コヒーレンスと照明 1 本、`BLIS_bf_*` BLIS-FPM の明視野全視野） |
| `01_データと前処理/` | 前処理・ドリフト・雑音・照明分類・較正の JSON |
| `02_明視野_512/`, `03_明視野_1024/` | DIP（最適反復 `DIP_*`、30 反復 `DIP30it_*`）と EPRY の TIFF・summary、比較の数値 |
| `04_BLIS-FPM_明視野/` | BLIS-FPM 全視野（9 枚）の TIFF・瞳・較正・帯域チェック・視野効果の走査・物体帯域制限の検証 |
| `05_暗視野_512/`, `06_暗視野_1024/` | 28 枚（リング 1 + 3）の DIP・DIP 30 反復・EPRY・BLIS-FPM（`BLIS_df_*`、全視野）の TIFF と比較の数値 |
| `07_暗視野_診断/` | BLIS-FPM 44 枚・28 枚・k/瞳テスト、EPRY の方式比較、半分平均の FRC |
| `09_部分コヒーレンス/` | 5 章の数値（`pc_summary.json` ほか：直接光の曲線、S の推定、予測テスト、モード数、合成データ、比較の数値 `compare_summary_*.json`、BLIS-FPM の解 `blis_*.json`）。全視野 1024 画素・28 枚の TIFF（部分コヒーレント `tiff_1024_partial_coherent/`、照明 1 本 `tiff_1024_1mode/`） |
| `08_パイプライン/` | 変更後の 3 パイプライン（036 用設定を含む）と 036 用スクリプトの zip。`FPT036_暗視野対応パイプライン_20260926.zip` は、これらを 1 つにまとめ、再現手順の README を付けたもの |

TIFF はすべて float32、画素 31.9 nm、物理フレーム（位相の符号 −1）。`*_transmission.tif`・`*_phase_rad.tif` は 0.3–3.5 µm⁻¹ に帯域制限、`*_unfiltered.tif` は帯域制限なし。
"""


def build():
    if os.path.exists(B): shutil.rmtree(B)
    os.makedirs(B + "figures")
    for dst, src in FIGS.items():
        if os.path.exists(W + src): shutil.copy2(W + src, B + "figures/" + dst)
        else: print("missing figure", src)
    for folder, items in COPY.items():
        os.makedirs(B + folder, exist_ok=True)
        for src, new in items:
            if os.path.exists(src): shutil.copy2(src, B + folder + "/" + (new or os.path.basename(src)))
            else: print("missing", src)
    os.makedirs(B + "08_パイプライン", exist_ok=True)
    for zname, (root, pats) in PIPE.items():
        with zipfile.ZipFile(B + "08_パイプライン/" + zname, "w", zipfile.ZIP_DEFLATED) as z:
            for pat in pats:
                for p in sorted(glob.glob(W + root + "/" + pat)):
                    z.write(p, root + "/" + os.path.relpath(p, W + root))
    import subprocess
    subprocess.run([sys.executable, W + "d036/make_pipeline_zip_036.py", B + "08_パイプライン/FPT036_暗視野対応パイプライン_20260926.zip"], check=True)
    rep = open(W + "d036/036_report.md", encoding="utf-8").read()
    rep = re.sub(r"\]\(([^)]+\.png)\)", lambda m: f"](figures/{LINKS.get(m.group(1), os.path.basename(m.group(1)))})", rep)
    rep = rep.replace("`out_512/fig_compare_planewave.png`", "`figures/bf_512_compare_planewave.png`").replace(
        "`out_1024/fig_compare_planewave.png`", "`figures/bf_1024_compare_planewave.png`")
    rep = rep.split("## ファイル\n")[0] + FILES_MD
    open(B + "036_report.md", "w", encoding="utf-8").write(rep)
    shutil.copy2(W + "d036/deck/FPT036_再構成まとめ.pptx", B)
    shutil.copy2(W + "d036/036_summary.json", B)
    missing = [m for m in re.findall(r"\]\(figures/([^)]+)\)", rep) if not os.path.exists(B + "figures/" + m)]
    n = sum(len(fs) for _, _, fs in os.walk(B)); size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(B) for f in fs)
    print(f"{n} files, {size / 1e6:.1f} MB; missing figure refs: {missing}")


if __name__ == "__main__":
    build()
