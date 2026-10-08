# -*- coding: utf-8 -*-
"""README of supplement_focus_in_pupil/ (Japanese): the posaffine focus loop (residual object defocus moved into the pupil)
and the re-solved 32 reconstructions. All numbers are read from the JSONs through make_report_32a.py.
usage (workspace root): python d32a/frc/make_readme_focus_in_pupil.py -> d32a/frc/README_focus_in_pupil_32a.md"""
import os
W = "./"
ns = {"__file__": os.path.abspath(W + "d32a/make_report_32a.py")}; exec(open(W + "d32a/make_report_32a.py", encoding="utf-8").read(), ns)
g = ns.get
C, M, ML, MOD, r, f2, f3 = g("C"), g("M"), g("ML"), g("MOD"), g("r"), g("f2"), g("f3")
MAIN, PREV, PAR, PAR1, RES2, PBT, J = g("MAIN"), g("PREV"), g("PAR"), g("PAR1"), g("RES2"), g("PBT"), g("J")
assert g("AFF2"), "aff2 results missing"
AQ = J("d32a/af_test/pupil_annulus_quadratic.json")
L0 = PAR["focus_loop"]; ck1, ck2 = PAR1["check"], PAR["check"]; mm1, mm2 = PAR1["mechanical"], PAR["mechanical"]
T = []; A = T.append
A("# 32a：物体側に残ったピントずれを瞳に移して解き直す（フォーカスループ、2026-09-28）")
A("")
A("このフォルダは、レポート（`../32a_report.md`）の 2.4 節と 7.3 節の補足です。図・JSON・スクリプトをまとめました。")
A("")
A("## 1. 何をしたか")
A(f"前の版（2026-09-27 版）では、瞳を「コントラストのピント」z = {PAR1['pupil_defocus_um']/1e3:+.2f} mm に決め、その瞳が予測する視差だけをデータに残していました。"
  f"ところが、この瞳で解いた EPRY・BLIS-FPM の物体には、約 +1.1 mm のピントずれが残っていました（明視野・512 px・FOV で EPRY {r(PREV['bf512'], 'EPRY', 'FOV model')['pupil_drift_check']['object_refocus']['z_um']/1e3:+.2f} mm、"
  f"BLIS-FPM {r(PREV['bf512'], 'BLIS', 'FOV model')['pupil_drift_check']['object_refocus']['z_um']/1e3:+.2f} mm）。")
A("そこで posaffine ステップに「フォーカスループ」（`position_affine.focus_iter`、`focus_tol_um`）を加えました。")
A("")
A("1. 瞳をコントラストのピント（デフォーカス＋非点だけの滑らかな瞳、`analytic_W`）に固定し、その視差を残したデータで物体だけを解く（`run_nonlinear(..., fix_pupil=True)`）")
A("2. 物体のピントを合わせる（`autofocus`）。再合焦量が 0.1 mm 以上なら、それを瞳に移し、残す視差と機械的な補正を計算し直して 1 に戻る")
A("")
A("このあと、4 手法 × 2 モデル × 2 格子 × 明視野／暗視野を含む、の全 32 の再構成と、分解能の評価用の半分ずつのデータの再構成（FRC 用）を解き直しました。")
A("")
A("## 2. フォーカスループの結果")
A("")
A("| くり返し | 固定した瞳 z / 非点 (mm) | 物体の再合焦 z (mm) | misfit |")
A("|---|---|---|---|")
for i_, l_ in enumerate(L0):
    A(f"| {i_} | {l_['pupil_defocus_um']/1e3:+.2f} / ({l_['pupil_astig_um'][0]/1e3:+.2f}, {l_['pupil_astig_um'][1]/1e3:+.2f}) | {l_['object_refocus']['z_um']/1e3:+.2f} | {l_['misfit']:.4f} |")
A("")
A(f"2 回で収束しました（瞳 z {PAR['pupil_defocus_um']/1e3:+.2f} mm）。misfit はほとんど変わりません。")
A("")
A("| 機械的な比例ずれ（%） | ループ前 | ループ後 |")
A("|---|---|---|")
for k_, lb_ in (("scale_col_pct", "列 ← X"), ("scale_row_pct", "行 ← Y"), ("col_from_row_axis_pct", "列 ← Y"), ("row_from_col_axis_pct", "行 ← X"), ("isotropic_pct", "等方成分"), ("rotation_deg", "回転（°）"), ("shear_pct", "せん断")):
    A(f"| {lb_} | {mm1[k_]:+.3f} | {mm2[k_]:+.3f} |")
A("")
A("**縮退について**：物体側のデフォーカス z は、「瞳のデフォーカス z」と「その視差（照明の位置 k_n に比例する像のずれ λzk_n）を打ち消す像のずれ」を組み合わせたものと同じ像を作ります。"
  "k_n に比例する像のずれは、機械的な比例ずれの等方成分（倍率）と区別できません。つまり、データは「物体のピントずれ」と「瞳のピントずれ＋倍率」を区別しません（misfit が変わらないのはこのためです）。"
  f"フォーカスループは「試料（ジーメンススター）にピントが合っている」ことを条件にして、この縮退を解いています。その分、機械的な倍率が {mm2['isotropic_pct'] - mm1['isotropic_pct']:+.2f} % 変わり、回転とせん断は変わりません。"
  "瞳のデフォーカスと倍率を独立に決めるには、試料を光軸方向に既知の量（例えば ±2 mm）動かした測定が有効です。")
A("")
zq = [f_["z_um"] / 1e3 for f_ in AQ["aff2"]["annulus_quadratic"]]
A(f"**瞳を自由に動かしたときの確認**（posaffine の確認の計算、BLIS-FPM）：物体の再合焦量は {ck1['object_refocus']['z_um']/1e3:+.2f} mm（ループ前）→ {ck2['object_refocus']['z_um']/1e3:+.2f} mm（ループ後）で、ループ後の物体はピントが合っています。"
  f"自由な瞳のデフォーカスは、ループの前後で {ck1['pupil']['defocus_um']/1e3:+.2f} → {ck2['pupil']['defocus_um']/1e3:+.2f} mm（posaffine の 4 次多項式）と、物体の再合焦量と同じだけ入れ替わりました。"
  f"ただし、自由な瞳は中心（|k| < 0.2 µm⁻¹）に約 {AQ['aff2']['central_step_rad']:.1f} rad の位相の段差をもち、動径方向の形も 2 次曲線からずれています。"
  f"そのため、デフォーカスの値は当てはめ方で変わり（4 次多項式 {ck2['pupil']['defocus_um']/1e3:+.2f} mm、回転対称な 2 次だけ {min(zq):+.2f}〜{max(zq):+.2f} mm、Zernike のバランスしたデフォーカス {PBT['aff2']['zernike']['z_balanced_um']/1e3:+.2f} mm）、"
  f"固定した瞳（{PAR['pupil_defocus_um']/1e3:+.2f} mm）より小さく出ます。posaffine の確認の値「(瞳 − 物体の再合焦) の比例ずれ − 残したずれ」が {ck2['total_focus_drift_minus_retained_pct']:.3f} % と基準（0.1 %）を超えたのは、このためです。"
  f"モデル像とデータの残りの比例ずれは {ck2['residual_linear_drift_pct']:.3f} % で、像の位置は合っています。")
A("")
A("![focus loop](fig_focusloop_32a.png)")
A("")
A("*図 1　(a) フォーカスループ。(b) 各再構成の物体の再合焦量（中央 512 px、FOV モデル）。(c) 各再構成の瞳のデフォーカス（破線：残した視差に相当するデフォーカス）。(d) 自由な瞳の動径方向の位相。*")
A("")
A("## 3. 再構成像（ループの前後）")
A("")
A("FOV モデル、帯域フィルターなし、再構成したまま（後から再合焦していない）の位相です。同じデータ（明視野、または暗視野を含む）の中では、ループの前後と 4 手法で共通のグレースケールにしています。")
A("")
for f_, cap_ in (("fig_focusloop_zoom_32a_512.png", "星の中心（5.1 µm 角）、中央 512 px の再構成"), ("fig_focusloop_field_32a_512.png", "中央 512 px 全体"),
                 ("fig_focusloop_zoom_32a_1000.png", "星の中心（5.1 µm 角）、全視野 1000 px の再構成"), ("fig_focusloop_field_32a_1000.png", "全視野 1000 px 全体")):
    A(f"![{cap_}]({f_})"); A(""); A(f"*{cap_}。上 2 段：明視野（ループ前 / 後）、下 2 段：暗視野を含む（ループ前 / 後）。破線の枠：ループ前、実線の枠：ループ後。数字はスポーク分解能・物体の再合焦量・その手法が求めた瞳のデフォーカス。段の見出しの瞳 z は全手法に共通の初期値（EPRY はそのまま、DIP・BLIS-FPM は更新）。*"); A("")
A("## 4. 解き直した 32 の再構成（ループ前 → 後、FOV モデル）")
A("")
A("| データ | 手法 | 外した画像の misfit | スポーク (µm⁻¹) | 物体の再合焦 (mm) | 瞳 z (mm) |")
A("|---|---|---|---|---|---|")
for gk, gl in (("bf512", "明視野・512 px"), ("bf1000", "明視野・1000 px"), ("df512", "暗視野を含む・512 px"), ("df1000", "暗視野を含む・1000 px")):
    for m, mn in M:
        xo, xn = r(PREV[gk], m, "FOV model"), r(MAIN[gk], m, "FOV model")
        po, pn = xo["pupil_drift_check"], xn["pupil_drift_check"]
        A(f"| {gl} | {mn} | {f3(xo['misfit_unfiltered']['heldout']['all'])} → {f3(xn['misfit_unfiltered']['heldout']['all'])} | {f2(xo['spoke_snr3_um_inv'])} → {f2(xn['spoke_snr3_um_inv'])} | "
          f"{po['object_refocus']['z_um']/1e3:+.2f} → {pn['object_refocus']['z_um']/1e3:+.2f} | {po['pupil_defocus_um']/1e3:+.2f} → {pn['pupil_defocus_um']/1e3:+.2f} |")
A("")
A("平面波モデルを含む全条件の表は、レポートの 7.3 節にあります。")
A("")
A("## 5. 分解能（シーメンススター、中央 512 px、FOV モデル）")
A("")
A("| 手法 | データ | MTF 10 %：前 | 前（物体を後から再合焦） | 後 | FRC half-bit：前 | 前（再合焦） | 後 |")
A("|---|---|---|---|---|---|---|---|")
fq = lambda d_: "—" if not d_ else (("≥ " if d_["no_crossing_below_support_limit"] else "") + f"{d_['q']:.2f}")
for s_ in ("BF", "BF+DF"):
    for m, mn in M:
        t_ = RES2[f"MTF10|512|{s_}|{ML[m]}, FOV model"]; fr = lambda row_: RES2.get(f"FRC_halfbit|{row_}|{s_}|{ML[m]}")
        A(f"| {mn} | {s_} | {f2(t_['aff'])} | {f2(t_['aff_refocused'])} | {f2(t_['aff2'])} | {fq(fr('as reconstructed')['aff'])} | {fq(fr('after object refocus')['aff'])} | {fq(fr('as reconstructed')['aff2'])} |")
A("")
A("（単位 µm⁻¹。「≥」は支持域の限界 3.75 µm⁻¹（明視野）までに交点がないことを示します。）")
A("")
for t_ in g("FL_TXT") or []:
    A(t_); A("")
A("![MTF 512](fig_mtf_signed_aff_vs_aff2_512.png)")
A("")
A("*図 2　符号つきの星の MTF（中央 512 px、FOV モデル）。灰色の破線：ループ前、色の点線：ループ前の物体を後から再合焦、色の実線：ループ後。帯は雑音 ±3σ。全視野 1000 px は `fig_mtf_signed_aff_vs_aff2_1000.png`。*")
A("")
A("![FRC 512](fig_frc_aff_vs_aff2_512.png)")
A("")
A("*図 3　半分ずつのデータから独立に解いた 2 つの再構成の FRC（位相）。線の種類は図 2 と同じ。支持域の限界より上（薄い線）は分解能の意味がありません。*")
A("")
A("## 6. ファイル")
A("- `position_affine_aff.json` / `position_affine_aff2.json`：posaffine の結果（ループ前 / 後。`focus_loop` にくり返しの値）")
A("- `focusloop_summary_32a.json`：図 1 の値（各再構成の再合焦量と瞳のデフォーカス、自由な瞳の動径方向の位相）")
A("- `pupil_basis_test.json`、`pupil_annulus_quadratic.json`：自由な瞳のデフォーカスを、当てはめ方を変えて読み取った値")
A("- `frc_mtf_32a_aff2.json`、`resolution_table_32a_aff2.json`：符号つき MTF と FRC（ループ後）と、前後の比較表")
A("- `fig_focusloop_{zoom,field}_32a_{512,1000}.png`（`fig_focusloop_images_32a.py`）：ループの前後の再構成像。値は `fig_focusloop_images_32a_{512,1000}.json`")
A("- `scripts/`：`fig_focusloop_32a.py`、`fig_focusloop_images_32a.py`、`pupil_basis_test.py`、`pupil_annulus_quadratic.py`、`frc_mtf_32a.py`（`aff2` を引数に）、`fig_resolution_aff2.py`、`setup_halves.py`（`aff2`）、"
  "`make_df_aff_32a.py`（`fpt_output_32a_bf_aff2 work_df_aff2`）、実行したシェル（`chainF2_bfaff2.sh`、`chainG2_dfaff2.sh`、`post_aff2.sh`、`chainH*x_*_aff2.sh`）")
A("- パイプラインの変更：`../pipeline_changes/fptrecon/posaffine.py`（フォーカスループ）、`nonlinear.py`（`fix_pupil`）、`config_32a_bf_aff2.json`。パイプラインのフォルダにも入れ、変更前は `*.bak_20260928`")
TXT = "\n".join(T) + "\n"
for a_, b_ in (("(-0.00,", "(0.00,"), (", -0.00)", ", 0.00)"), ("(+0.00,", "(0.00,"), (", +0.00)", ", 0.00)")): TXT = TXT.replace(a_, b_)
open(W + "d32a/frc/README_focus_in_pupil_32a.md", "w", encoding="utf-8").write(TXT)
print("lines", len(T))
