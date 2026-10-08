# -*- coding: utf-8 -*-
"""FPT #036 report (Japanese markdown) and machine-readable summary, generated from the analysis JSONs.
usage: python d036/make_report_036.py      -> d036/036_report.md, d036/036_summary.json"""
import json, os
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
J = lambda p: json.load(open(W + p, encoding="utf-8"))
ex = lambda p: os.path.exists(W + p)
f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"
ring_of = lambda i: 1 if i < 9 else (2 if i < 25 else 3)

prep = J("d036/data/prep_036.json"); prepdf = J("d036/work_df/prep_df.json"); noise = J("d036/work_df/noise.json")
cal = J("fpt_pipeline/fpt_output_036bf/work/calibration.json"); ill = J("d036/data/illumination_036.json")["positions"]
sP = J("fpt_pipeline/fpt_output_036bf/results/planewave_summary.json"); sF = J("fpt_pipeline/fpt_output_036bf/results/FOV_cp1_00_summary.json")
bcP = J("fpt_pipeline/fpt_output_036bf/work/nonlinear_cp0_00_bandcheck.json"); bcF = J("fpt_pipeline/fpt_output_036bf/work/nonlinear_cp1_00_bandcheck.json")
scan = J("d036/kappa_scan/scan.json"); kdir = J("d036/data/kappa_direct_fit.json")["best"]
C = {t: J(f"d036/out_{t}/compare_summary.json") for t in ("512", "1024", "512_objband", "512_objband_ho", "512_df", "512_df_ho", "512_df_bp")}
if ex("d036/out_1024_df/compare_summary.json"): C["1024_df"] = J("d036/out_1024_df/compare_summary.json")
if ex("d036/out_1024_df_ho/compare_summary.json"): C["1024_df_ho"] = J("d036/out_1024_df_ho/compare_summary.json")
C18 = {t: J(f"d018/out_{t}/compare_summary.json") for t in ("512", "1024")}
b44 = {k: J(f"d036/blis_df/{k}.json") for k in ("s-1", "s-1_ho")}
b13 = {k: J(f"d036/blis_df13/{k}.json") for k in ("s-1", "s-1_ho", "s+0", "s+0_ho")}
kt = {k: J(f"d036/blis_k3test/s-1_ho_j100{k}.json") for k in ("", "_k3x0.97", "_k3x1.03", "_P0.98", "_P1.06")}
frc = J("d036/frc_halves.json")
ediag = {v: J(f"d036/epry_diag/{v}.json") for v in ("collapse", "firstvisit", "brightonly", "r13", "r13fix")}
illum = J("d036/data/illum_level.json")
M = ["DIP (optimal it.)", "DIP (30 it.)", "EPRY", "BLIS-FPM"]
ML = {"DIP (optimal it.)": "DIP（最適反復）", "DIP (30 it.)": "DIP（30 反復）", "EPRY": "EPRY", "BLIS-FPM": "BLIS-FPM"}
MOD = {"plane-wave model": "平面波", "FOV model": "FOV"}
r = lambda t, m, mod: C[t]["results"][f"{m}, {mod}"]

def per_ring(t, m, mod):
    s = C[t]; v = r(t, m, mod); tr, va = s["image_index_train"], s["image_index_validation"]
    pt, pv = np.array(v["band_misfit_training_per_image"]), np.array(v["band_misfit_validation_per_image"])
    out = {}
    for R in (1, 3):
        a = pt[[ring_of(i) == R for i in tr]]; b = pv[[ring_of(i) == R for i in va]]
        out[f"fit{R}"] = float(a.mean()) if len(a) else float("nan"); out[f"ho{R}"] = float(b.mean()) if len(b) else float("nan")
    return out

# ---------------------------------------------------------------- numbers used in the text
ring = np.array([p["ring"] for p in ill]); kk = np.array([p["k_over_kc"] for p in ill]); cover = np.array([p["direct_cover"] for p in ill])
lev = {R: np.median([p["level"] for p in prepdf["positions"] if p["ring"] == R]) for R in (2, 3)}
nf512 = {R: np.median([p["noise_frac_512"] for p in prepdf["positions"] if p["ring"] == R]) for R in (1, 2, 3)}
nf1024 = {R: np.median([p["noise_frac_1024"] for p in prepdf["positions"] if p["ring"] == R]) for R in (1, 2, 3)}
ill2 = [o["p99"] for o in illum if o["ring"] == 2]; ill1 = [o["p99"] for o in illum if o["ring"] == 1]
drift = np.array(prep["drift_end_px"]) * 0.0319
scan_c = sorted(((-v["s_solver"], v["misfit"]) for v in scan.values()), key=lambda t: t[0])
cbest = min(scan_c, key=lambda t: t[1])
spoke1024 = {f"{m}, {mod}": r("1024", m, mod)["spoke_snr3_um_inv"] for m in M for mod in MOD}
frcL = {k: v["limit_1_7"] for k, v in frc.items()}
def ed_min(v, key):
    h = ediag[v]["hist"]; return min(x[key] for x in h)
def ed_at_best(v):
    h = ediag[v]["hist"]; i = int(np.argmin([x["ring1_heldout"] + x["ring3_heldout"] for x in h])); return h[i]

summary = dict(dataset="036", conditions=dict(energy_keV=30.0, lam_nm=0.041328, pixel_nm=31.9, fzp_dr_nm=200, fzp_diam_um=155, kc_um_inv=2.5,
                                                p_m=0.75, czp_f_m=2.4, um_per_pulse=0.5, urad_per_pulse_nominal=0.20833),
               preprocessing=dict(hot_pixel_fraction=prep["hot_pixel_fraction"], background_median=prep["background_median"],
                                  drift_end_um=drift.tolist(), drift_fit_rms_px=prep["drift_fit_residual_rms_px"],
                                  darkfield_level_ring2=lev[2], darkfield_level_ring3=lev[3], noise_fraction_512=nf512, noise_fraction_1024=nf1024),
               calibration={k: cal[k] for k in ("rotation_deg", "scale_vs_nominal", "angle_per_pulse_urad", "defocus_um", "astig_um", "misfit", "corr_a_phi_linear", "twin")},
               blis_bf=dict(plane=sP["final_misfit"], fov=sF["final_misfit"], defocus_plane_um=sP["defocus_equiv_um"], defocus_fov_um=sF["defocus_equiv_um"],
                            bandcheck_plane=bcP, bandcheck_fov=bcF),
               fov_scan_c_misfit=scan_c, fov_direct_image_fit=kdir,
               phaseB={t: {k: {kk_: v[kk_] for kk_ in ("band_misfit_training_images", "band_misfit_validation_images", "spoke_snr3_um_inv", "phase_rms_band", "iteration") if kk_ in v}
                           for k, v in C[t]["results"].items()} for t in ("512", "1024", "512_objband", "512_objband_ho")},
               phaseC={t: {f"{m}, {mod}": per_ring(t, m, mod) for m in M for mod in MOD} for t in C if "df" in t},
               blis_44=b44, blis_13=b13, blis_ring3_tests={k or "reference": v["heldout_per_image"] for k, v in kt.items()},
               frc_halves_limit_1_7=frcL, epry_dark_variants={v: ediag[v]["hist"][-1] for v in ediag})
json.dump(summary, open(W + "d036/036_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)

# ---------------------------------------------------------------- tables
def tab_B(t, ho_note):
    rows = ["| 手法 | 平面波：学習 | 平面波：未使用 | FOV：学習 | FOV：未使用 | 選ばれた反復 (平面波 / FOV) |", "|---|---|---|---|---|---|"]
    for m in M:
        a, b = r(t, m, "plane-wave model"), r(t, m, "FOV model")
        it = f"{a.get('iteration')} / {b.get('iteration')}" if a.get("iteration") is not None else "—"
        star = "＊" if m == "BLIS-FPM" else ""
        rows.append(f"| {ML[m]} | {f3(a['band_misfit_training_images'])} | {f3(a['band_misfit_validation_images'])}{star} | "
                    f"{f3(b['band_misfit_training_images'])} | {f3(b['band_misfit_validation_images'])}{star} | {it} |")
    return "\n".join(rows) + f"\n\n＊ {ho_note}"
def tab_C(t, tho):
    rows = ["| 手法 | モデル | リング 1：学習 | リング 1：未使用 | リング 3：学習 | リング 3：未使用 |", "|---|---|---|---|---|---|"]
    for m in M:
        for mod in MOD:
            v = per_ring(tho if (m == "BLIS-FPM" and tho) else t, m, mod)
            rows.append(f"| {ML[m]} | {MOD[mod]} | {f3(v['fit1'])} | {f3(v['ho1'])} | {f3(v['fit3'])} | {f3(v['ho3'])} |")
    return "\n".join(rows)
bo, bh = C["512_objband"]["results"], C["512_objband_ho"]["results"]
pB5 = lambda m, mod, k="band_misfit_validation_images": r("512", m, mod)[k]
pB10 = lambda m, mod, k="band_misfit_validation_images": r("1024", m, mod)[k]

L = []; A = L.append
A("# FPT #036 シーメンススター：DIP・EPRY・BLIS-FPM の比較（明視野のみ → 暗視野を含む場合）\n")
A("データ：036（30 keV、44 照明位置 × 試料あり/なし × 6 回）。生成日 2026-09-26。数値はすべて解析結果の JSON（`036_summary.json` ほか）から自動で書き出しています。\n")
A("## 要点\n")
A(f"1. **前処理**：6 回の撮影のあいだに試料が {abs(drift[0]):.2f} µm（縦）・{abs(drift[1]):.2f} µm（横）ずれていたため、明視野の比画像の相互相関でずれを求め（3 次式で近似、残差 {prep['drift_fit_residual_rms_px'][0]:.2f} / {prep['drift_fit_residual_rms_px'][1]:.2f} px）、位置をそろえてから中央値で平均しました。検出器の画素の {prep['hot_pixel_fraction'] * 100:.0f} % がホット画素（暗レベル > 250 counts）で、周囲の正常画素の平均で置き換えています。")
A(f"2. **照明の分類**：リング 1（9 位置、|k|/k_c = {kk[ring == 1].min():.2f}–{kk[ring == 1].max():.2f}）は明視野、リング 2（16 位置、{kk[ring == 2].min():.2f}–{kk[ring == 2].max():.2f}）は視野の端だけ直接光が入る境界、リング 3（19 位置、{kk[ring == 3].min():.2f}–{kk[ring == 3].max():.2f}）は暗視野です。較正はリング 1 だけで行い（misfit {f3(cal['misfit'])}）、リング 2・3 の k はステージ座標からの変換で与えました。")
A(f"3. **明視野のみ（9 枚）の比較**：どの手法でも **FOV モデルの方がはっきり良く**、未使用画像の misfit は中央 512 画素で DIP（最適反復） {f3(pB5('DIP (optimal it.)', 'plane-wave model'))} → {f3(pB5('DIP (optimal it.)', 'FOV model'))}、EPRY {f3(pB5('EPRY', 'plane-wave model'))} → {f3(pB5('EPRY', 'FOV model'))}、BLIS-FPM（2 枚を除いて再計算） {f3(bh['BLIS-FPM, plane-wave model']['band_misfit_validation_images'])} → {f3(bh['BLIS-FPM, FOV model']['band_misfit_validation_images'])} でした（018 では DIP・EPRY で差がありませんでした）。未使用画像に最もよく合うのは DIP（最適反復）・FOV で、512 画素 {f3(pB5('DIP (optimal it.)', 'FOV model'))}、1024 画素 {f3(pB10('DIP (optimal it.)', 'FOV model'))} です。")
A(f"4. **分解能（明視野のみ）**：全視野 1024 画素のスポーク変調（SNR > 3）は 4 手法とも {min(spoke1024.values()):.2f}–{max(spoke1024.values()):.2f} µm⁻¹（半周期 {1e3 / (2 * max(spoke1024.values())):.0f}–{1e3 / (2 * min(spoke1024.values())):.0f} nm）で、対物の k_c = 2.5 µm⁻¹ の約 1.3–1.4 倍です。データの信号が雑音と同じになる約 3.0–3.4 µm⁻¹ で決まっています。")
A(f"5. **暗視野への拡張**：3 つのパイプラインに暗視野用の処理を追加しました（既定の動作は不変）。リング 2 は、部分的にコヒーレントな直接光の裾（中央で明視野の 2–5 %）と縁の干渉縞が像を支配し、コヒーレントな順モデルで表せないため除外しました。リング 1 + 3 の 28 枚では、学習に使ったリング 3 の画像（中央 512 画素、FOV）に BLIS-FPM はよく合い（{f3(per_ring('512_df', 'BLIS-FPM', 'FOV model')['fit3'])}）、DIP・EPRY はあまり合いませんでした（DIP 最適 {f3(per_ring('512_df', 'DIP (optimal it.)', 'FOV model')['fit3'])}、DIP 30 反復 {f3(per_ring('512_df', 'DIP (30 it.)', 'FOV model')['fit3'])}、EPRY {f3(per_ring('512_df', 'EPRY', 'FOV model')['fit3'])}）。**使っていないリング 3 の画像は、中央 512 画素では 4 手法とも予測できませんでした**（未使用のリング 3 の misfit：DIP 最適 {f3(per_ring('512_df', 'DIP (optimal it.)', 'FOV model')['ho3'])}、EPRY {f3(per_ring('512_df', 'EPRY', 'FOV model')['ho3'])}、BLIS-FPM {f3(per_ring('512_df_ho', 'BLIS-FPM', 'FOV model')['ho3'])}；1.0 は「何も予測しない」のと同じ）。k を ±3 %、瞳の半径を ±4 % 変えても 1.1 を下回りません。全視野 1024 画素（FOV）では、未使用のリング 3 の misfit は DIP 最適 {per_ring('1024_df', 'DIP (optimal it.)', 'FOV model')['ho3']:.2f}、DIP 30 反復 {per_ring('1024_df', 'DIP (30 it.)', 'FOV model')['ho3']:.2f}、EPRY {per_ring('1024_df', 'EPRY', 'FOV model')['ho3']:.2f}、BLIS-FPM（4 枚を除いた解）{per_ring('1024_df_ho', 'BLIS-FPM', 'FOV model')['ho3']:.2f} でした。DIP・EPRY はわずかに 1 を下回りますが、暗視野像の強度を予測できていると言える値ではありません。")
spd = {m: C["1024_df"]["results"][m + ", FOV model"]["spoke_snr3_um_inv"] for m in M} if "1024_df" in C else {}
spdh = C["1024_df_ho"]["results"]["BLIS-FPM, FOV model"]["spoke_snr3_um_inv"] if "1024_df_ho" in C else float("nan")
A(f"6. **超解像の評価**：全視野のスポーク変調（SNR > 3）は、暗視野を加えると BLIS-FPM で {spoke1024['BLIS-FPM, FOV model']:.2f} → {spd['BLIS-FPM']:.2f} µm⁻¹（4 枚を除いて解いても {spdh:.2f}；半周期 {1e3 / (2 * spoke1024['BLIS-FPM, FOV model']):.0f} → {1e3 / (2 * spd['BLIS-FPM']):.0f} nm、k_c の {spd['BLIS-FPM'] / 2.5:.1f} 倍）、DIP（最適反復）で {spoke1024['DIP (optimal it.), FOV model']:.2f} → {spd['DIP (optimal it.)']:.2f}、EPRY で {spoke1024['EPRY, FOV model']:.2f} → {spd['EPRY']:.2f} µm⁻¹ に上がりました（FOV モデル）。半分平均どうしの位相 FRC も延びます（EPRY {frcL['EPRY, bright field (ring 1), centre 512']:.1f} → {frcL['EPRY, bright + dark field (rings 1, 3), centre 512']:.1f}、BLIS-FPM {frcL['BLIS-FPM, bright field (ring 1), full field']:.1f} → {frcL['BLIS-FPM, bright + dark field (rings 1, 3), full field']:.1f} µm⁻¹）。**暗視野画像は星の細いスポークの情報を実際に含み、BLIS-FPM が最もよく取り込んでいます**。ただし 5 のとおり暗視野像の強度そのものは予測できていないので、高周波成分の振幅や細部までは確かめられていません。")
_PSq = json.load(open(W + "d036/pc/pc_summary.json")) if os.path.exists(W + "d036/pc/pc_summary.json") else None
if _PSq:
    _tq = _PSq["tables"]; _gq = lambda t, m, k="ho1": _tq[t]["results"][f"{m}, FOV model"][k]
    _iq = [100 * (1 - _gq(t + "_pc", m) / _gq(t + "_pc1m", m)) for t in ("512", "1024") for m in M]
    _h3 = [_gq(t + "_pc", m, "ho3") for t in ("512", "1024") for m in M]
    A(f"7. **部分コヒーレンス（5 章）**：試料なし画像の直接光が瞳の縁の外へゆっくり減衰することから照明の角度広がりを見積もり、照明を約 20 本の平面波のインコヒーレントな和として 3 つのパイプラインに入れました。すその割合 η = {_PSq['eta']} で、暗視野を含む 28 枚の計算では、未使用の明視野画像の misfit が照明 1 本に比べ 4 手法とも {min(_iq):.0f}–{max(_iq):.0f} % 下がりました（明視野だけの 9 枚の BLIS-FPM では予測は良くならず）。一方、未使用のリング 3 の画像は {min(_h3):.2f}–{max(_h3):.2f} で予測できないまま、スポークの分解能も EPRY 以外は変わりませんでした（EPRY の違いは、選ばれた掃引の回数が照明 1 本で {_tq['1024_pc1m']['results']['EPRY, FOV model']['iteration']}、部分コヒーレントで {_tq['1024_pc']['results']['EPRY, FOV model']['iteration']} と違うため）。リング 3 の予測を妨げているのは照明の角度広がりではないと考えられます。")
A("")
A("## 1. 測定条件とデータ\n")
A("| 項目 | 値 |\n|---|---|")
for k_, v_ in [("X 線エネルギー", "30.0 keV（λ = 0.041328 nm）"), ("実効画素", "31.9 nm（1024 × 1024、32.7 µm 角）"),
               ("対物 FZP", "Δr = 200 nm、φ155 µm → NA = 1.033 × 10⁻⁴、k_c = 2.50 µm⁻¹"), ("試料–対物距離 p", "0.75 m（FZP 焦点距離。実測ではない）"),
               ("コンデンサ CZP", "f = 2.4 m（実測ではない）、0.5 µm/pulse → 公称 0.20833 µrad/pulse"),
               ("走査", "3層高エネczp.csv の 44 点（リング 1/2/3 = 9/16/19 点）"),
               ("撮影順", "44 枚ごとに 試料あり → 試料なし を 6 回（a001–a528、各 1 s）"),
               ("ヘッダのオフセット", "ConversionFactorOffset = 100")]:
    A(f"| {k_} | {v_} |")
A("")
A(f"![データ](036_fig_data.png)\n\n**図 1** (a) 試料のずれ（明視野 9 位置 × 5 回の相互相関の測定点と 3 次式の近似）。(b) 44 照明の波数ベクトル（ソルバ座標）と対物の瞳 |k| = k_c。(c)–(f) 中央 512 画素：リング 1 の比画像 S/D、リング 2 の試料なし画像（直接光の裾）、リング 2 と 3 の S − D（明視野のレベルで規格化）。\n")
A("### 前処理\n")
A(f"- **背景**：リング 3（暗視野）の試料なし画像 19 × 6 枚の画素ごとの中央値を背景マップ B とし（中央値 {prep['background_median']:.0f} counts）、全フレームから差し引きました。オフセット 100 は B に含まれるので、パイプラインの dark_offset は 0 です。")
A(f"- **ホット画素**：B > 250 counts の画素（{prep['hot_pixel_fraction'] * 100:.1f} %）を、5 × 5 近傍の正常画素の平均で置き換えました。018 とは別の画素で、時間的には安定です。")
A(f"- **ドリフト補正**：リング 1 の比画像（S − B）/（D − B）を 1 回目と相互相関し、フレーム番号 t の 3 次式 d(t) で近似しました。終わりまでのずれは {prep['drift_end_px'][0]:+.1f} / {prep['drift_end_px'][1]:+.1f} px（{drift[0]:+.2f} / {drift[1]:+.2f} µm）です。試料ありのフレームだけを d(t) で戻し、試料なしのフレームは動かしていません。")
A("- **平均**：明視野の位置では 6 回の比画像の中央値、それ以外は位置をそろえたフレームの中央値です。奇数回（1, 3, 5 回目）と偶数回の半分平均 A・B も作り、パイプラインには n_repeat = 2 として渡しました（撮影 2 回の差から雑音を見積もるため）。")
A("")
A("## 2. 較正と照明の分類\n")
A(f"- **WOTF 較正**（リング 1 の 9 枚）：回転 {cal['rotation_deg']:+.2f}°、倍率は公称の {cal['scale_vs_nominal']:.3f} 倍（{cal['angle_per_pulse_urad']:.3f} µrad/pulse。003 の 1.31 倍に近い）、デフォーカス {cal['defocus_um']:.0f} µm、非点 ({cal['astig_um'][0]:.0f}, {cal['astig_um'][1]:.0f}) µm、misfit {f3(cal['misfit'])}。")
A(f"- **位相の符号（双対解）**：線形解の corr(a, φ) = {cal['corr_a_phi_linear']:+.3f} と強く、自動判定の −1 をそのまま使いました（018 と同じ符号。デフォーカスの符号も物理的）。")
A(f"- **照明の分類**：リング 2 の試料なし画像では、視野の端の {cover[ring == 2].min() * 100:.0f}–{cover[ring == 2].max() * 100:.0f} % に直接光が入ります。その明るさは明視野の {min(ill2):.2f}–{max(ill2):.2f} 倍（リング 1 の同じ指標は {min(ill1):.2f}–{max(ill1):.2f}）で、リング 2 の照明強度はリング 1 と同程度です。リング 3 では直接光は入りません。")
A(f"- **視野効果の強さ**：試料なし画像の明るい側はすべて照明の +k 方向にあり（位置によって局所的な照明角が変わる視野効果）、その境界の形からは c ≈ {kdir['c']:.2f}（瞳半径 {kdir['pupil_scale']:.2f} k_c のとき）が得られました。一方、BLIS-FPM の misfit（70 反復）を κ = c/(λp) で走査すると最小は c ≈ {cbest[0]:.1f} 付近でした（" + "、".join(f"c = {c_:g}: {m_:.3f}" for c_, m_ in scan_c) + "）。比較はすべて公称の c = 1 で行っています。")
A("")
A("## 3. 明視野のみ（リング 1、9 枚）の 4 手法比較\n")
A("018 と同じパイプライン・同じ評価指標（強度コントラストの 0.3–3.5 µm⁻¹ 帯域の相対誤差、物体は 0.3–3.5 µm⁻¹ に帯域制限してから順計算）です。未使用画像は 9 枚のうち 2 枚（画像 0 と 5）です。\n")
A("### 3.1 中央 512 画素\n")
A(tab_B("512", "BLIS-FPM は 9 枚すべてで解いているため、未使用列の値もその 2 枚を学習に使った値です。2 枚を除いて解き直した結果は下の表。"))
A("")
A(f"BLIS-FPM の追加検証（物体を |q| ≤ 3.5 µm⁻¹ に制限して最適化）：9 枚すべてで 平面波 {f3(bo['BLIS-FPM, plane-wave model']['band_misfit_training_images'])}、FOV {f3(bo['BLIS-FPM, FOV model']['band_misfit_training_images'])}。画像 0・5 を除いて解くと、その 2 枚の misfit は 平面波 {f3(bh['BLIS-FPM, plane-wave model']['band_misfit_validation_images'])}、FOV {f3(bh['BLIS-FPM, FOV model']['band_misfit_validation_images'])}（学習画像 {f3(bh['BLIS-FPM, plane-wave model']['band_misfit_training_images'])} / {f3(bh['BLIS-FPM, FOV model']['band_misfit_training_images'])}）。DIP（最適反復）の未使用画像 {f3(pB5('DIP (optimal it.)', 'plane-wave model'))} / {f3(pB5('DIP (optimal it.)', 'FOV model'))} より悪く、018 と同じく BLIS-FPM は学習画像への当てはめすぎの傾向があります（018 ほど大きくはない）。\n")
A(f"BLIS-FPM 全視野（9 枚、170 反復）の misfit は 平面波 {f3(sP['final_misfit'])}、FOV {f3(sF['final_misfit'])}（{(1 - sF['final_misfit'] / sP['final_misfit']) * 100:.0f} % 減）。FOV 解の位相パワーは {bcF['phase_power_fraction']['band'] * 100:.1f} % が 0.3–3.5 µm⁻¹ にあり、帯域制限しても {f3(bcF['object 0.3-3.5']['misfit'])}（平面波 {f3(bcP['object 0.3-3.5']['misfit'])}）なので、018 と違い **FOV モデルの改善は帯域外への逃げではありません**。\n")
A("![512 FOV](out_512/fig_compare_FOV_c1.png)\n\n**図 2** 明視野のみ・中央 512 画素・FOV モデル。上から DIP（最適反復）、DIP（30 反復）、EPRY、BLIS-FPM。下段：DIP（最適反復）との位相 FRC、瞳の Zernike 係数、misfit。平面波モデルは `out_512/fig_compare_planewave.png`。\n")
A("### 3.2 全視野 1024 画素\n")
A(tab_B("1024", "BLIS-FPM は 9 枚すべてを使った全視野解です（1024 の未使用画像検証は未実施）。"))
A("")
A("スポーク変調（シーメンススターの 36 本、SNR > 3 が続く最低の周波数、星の中心は (288, 525) 画素）：\n")
A("| 手法 | 平面波 | FOV |\n|---|---|---|")
for m in M:
    A(f"| {ML[m]} | {spoke1024[m + ', plane-wave model']:.2f} µm⁻¹ | {spoke1024[m + ', FOV model']:.2f} µm⁻¹ |")
A("\n星の中心は中央 512 画素の上端から約 1 µm の位置にあり、512 画素ではスポーク解析ができません（半径 30 画素以内しか取れない）。\n")
A("![1024 FOV](out_1024/fig_compare_FOV_c1.png)\n\n**図 3** 明視野のみ・全視野 1024 画素・FOV モデル（平面波は `out_1024/fig_compare_planewave.png`）。\n")
A("## 4. 暗視野への拡張（リング 1 + 3、28 枚）\n")
A("### 4.1 暗視野データの作り方とパイプラインの変更\n")
A(f"- **データ**：暗視野と境界の位置では試料なし画像で割れないので、各フレームで n = [ S − G₃{{D}} ] / ( g · U ) を作り、ドリフト補正して中央値で平均しました。G₃{{D}} は同じ位置の試料なし画像（σ = 3 px で平滑化）で直接光の裾と迷光を差し引くため、g は その回のリング 1 の直接光レベル、U はリング 1 の試料なし画像から求めた照明の包絡（σ = 30 px）です。値は明視野レベルを 1 とした共通のスケールで、リング 2 は約 {lev[2] * 100:.2f} %、リング 3 は約 {lev[3] * 100:.3f} % です。")
A(f"- **雑音**：半分平均の差から、0.3–3.5 µm⁻¹ の帯域で雑音が占める割合（中央 512 画素）はリング 1 / 2 / 3 で {nf512[1]:.3f} / {nf512[2]:.2f} / {nf512[3]:.2f} です。リング 3 は信号と雑音がほぼ同じ大きさです。")
A("- **順モデル**：|k_n| > 1.02 k_c の画像は暗視野として扱い、直接光による規格化をしません（O = 1 の明視野を 1 とした強度）。試料なし画像が明るい画素（明視野の 5–15 % 以上）は重みを下げて除外し、コントラストは除外後の平均に対して取ります。")
A("- **手法ごとの追加**（設定で切り替え、既定の動作＝003/018 の結果は変わりません）：")
A("  - DIP：暗視野画像の重みを雑音の逆数（σ²_明視野 / σ²_n）にし、照明強度の分からない暗視野画像の強度係数は最小二乗で毎回決める（`dark_gain: profile`）。")
gcol = [x["g_ring3"] for x in ediag["collapse"]["hist"]]
A(f"  - EPRY：暗視野画像の強度係数を掃引ごとに推定し直すと、係数が 0 に向かって縮み続けます（リング 3 で {gcol[0]:.2f} → {gcol[-1]:.3f}、{len(gcol)} 掃引）。係数が小さいほど暗視野画像の拘束が弱くなる、自己強化的な解です。そこで、各暗視野画像を最初に更新するときに 1 回だけ推定して固定する方式にしました。")
A("  - BLIS-FPM：画像ごとに雑音の逆数の重み（コントラスト単位）と有効画素のマスクを付けた損失。")
A("")
A("### 4.2 リング 2（境界）を除いた理由\n")
b44r = b44["s-1"]["mean_per_ring"]
A(f"44 枚すべてで BLIS-FPM（FOV）を解くと、リング 2 の misfit は平均 {b44r['2']:.2f}（雑音の割合 {nf1024[2]:.2f}）にとどまり、モデル像にはリング 2 のデータの主な構造（縁の明るい部分と同心円状の干渉縞）がほとんど現れません（図 4a）。リング 2 では、照明の角度の広がり（部分コヒーレンス）のため直接光の裾が瞳の中に入り、その裾で作られる明視野像と縁の回折縞が暗視野信号より大きくなっています。一つの照明角を仮定した順モデルでは表せないので、以下ではリング 2 を除いた 28 枚（リング 1 の 9 枚 + リング 3 の 19 枚）を使いました。未使用画像は 9 枚ごとの 4 枚（画像 0、25、34、43：明視野 1 枚 + 暗視野 3 枚）です。\n")
A("### 4.3 結果（中央 512 画素、28 枚）\n")
A("暗視野の画像は 3.5 µm⁻¹ より上の物体の成分で作られるので、この節の misfit は**帯域制限しない物体**から計算しています（強度の評価帯域は同じ 0.3–3.5 µm⁻¹）。BLIS-FPM の未使用列は 4 枚を除いて解き直した結果です。\n")
A(tab_C("512_df", "512_df_ho"))
A("")
A("- 1.0 は、その画像のコントラストをまったく予測しない（0 と予測する）場合の値です。")
A(f"- 未使用のリング 3 の画像は、4 手法ともおよそ 1 以上で、**予測できていません**。学習に使ったリング 3 の画像では BLIS-FPM が最も低く（{f3(per_ring('512_df', 'BLIS-FPM', 'FOV model')['fit3'])}、全視野では {b13['s-1']['mean_per_ring']['3']:.3f}）、雑音の割合（1024 で {nf1024[3]:.2f}）を下回るので、雑音まで当てはめています。")
A(f"- BLIS-FPM で、リング 3 の k を 0.97 / 1.00 / 1.03 倍にしたとき未使用リング 3 の misfit（全視野、100 反復）は "
  + " / ".join(f"{np.mean(kt[k]['heldout_per_image'][1:]):.2f}" for k in ("_k3x0.97", "", "_k3x1.03"))
  + f"、瞳の半径を 0.98 / 1.02 / 1.06 k_c にしたときは " + " / ".join(f"{np.mean(kt[k]['heldout_per_image'][1:]):.2f}" for k in ("_P0.98", "", "_P1.06")) + " で、いずれも 1 を超えます。k の外挿誤差や瞳の半径だけでは説明できません。")
ed = {v: ed_at_best(v) for v in ediag}
A(f"- EPRY（FOV、44 枚で評価）の方式の比較：明視野だけで解いた物体でリング 3 を予測すると misfit {ed['brightonly']['ring3_heldout']:.1f}、強度係数の推定がリング 3 で {ed['brightonly']['g_ring3']:.3f} まで下がります（物体に暗視野を作る高周波が足りない）。リング 1 + 3 で解いても（最初の訪問で係数を固定）未使用リング 3 は最良で {ed_min('r13', 'ring3_heldout'):.2f}、係数を 0.8 に固定すると {ed_min('r13fix', 'ring3_heldout'):.2f} でした。")
A(f"- DIP は、強度係数を学習パラメータ（初期値 1）のままにすると暗視野画像の係数がほとんど動かず（1.01）、未使用リング 3 の misfit が悪化しました（`*_v1gain`）。最小二乗で決める方式（上の表）で、選ばれる反復が 20 → 80（FOV）に延びました。")
A(f"- EPRY（FOV）は、未使用画像の misfit が {r('512_df', 'EPRY', 'FOV model')['iteration'] + 1} 回目（中央 512 画素）/ {r('1024_df', 'EPRY', 'FOV model')['iteration'] + 1} 回目（全視野）の掃引で最小になり、表の値と図はその時点の像です。位相の帯域内 rms は、明視野のみの {r('512', 'EPRY', 'FOV model')['phase_rms_band']:.3f} rad に対して {r('512_df', 'EPRY', 'FOV model')['phase_rms_band']:.3f} rad（中央 512 画素）と小さく、暗視野を加えた EPRY の像はほとんど反復が進んでいない状態です。")
if "1024_df" in C:
    A("")
    A("#### 全視野 1024 画素（28 枚）\n")
    A(tab_C("1024_df", "1024_df_ho" if "1024_df_ho" in C else None))
    A("")
    sp = {f"{m}, {mod}": r("1024_df", m, mod)["spoke_snr3_um_inv"] for m in M for mod in MOD}
    A("スポーク変調（SNR > 3）：" + "、".join(f"{ML[m]} {sp[m + ', plane-wave model']:.2f} / {sp[m + ', FOV model']:.2f}" for m in M) + " µm⁻¹（平面波 / FOV）。明視野のみでは " + "、".join(f"{ML[m]} {spoke1024[m + ', plane-wave model']:.2f} / {spoke1024[m + ', FOV model']:.2f}" for m in M) + " µm⁻¹。\n")
A("")
A("![暗視野](036_fig_darkfield.png)\n\n**図 4** (a) リング 2 の画像（データ）と BLIS-FPM（44 枚、FOV）のモデル像。(b) 学習に使ったリング 3 の画像とモデル像。(c) 使っていないリング 3 の画像と予測。いずれも全視野、0.3–3.5 µm⁻¹ の帯域のコントラスト。(d) FOV モデルの未使用画像の misfit（棒）と同じリングの学習画像の平均（点）、中央 512 画素。(e) 半分平均 A・B から別々に再構成した位相の FRC。灰色の帯はリング 1 だけでは物体の周波数が届かない範囲（|k| + k_c ≈ 1.83 k_c）で、この帯の中の FRC の上昇は窓関数と初期値に由来する見かけのものです。\n")
A("### 4.4 超解像の評価\n")
if "1024_df" in C:
    A("**スポーク変調**（全視野 1024 画素、FOV モデル、SNR > 3 が続く最低の周波数）：\n")
    A("| 手法 | リング 1 のみ（9 枚） | リング 1 + 3（28 枚） | 同、4 枚を除いた解 |\n|---|---|---|---|")
    for m in M:
        ho_ = f"{spdh:.2f} µm⁻¹" if m == "BLIS-FPM" else "—"
        A(f"| {ML[m]} | {spoke1024[m + ', FOV model']:.2f} µm⁻¹ | {spd[m]:.2f} µm⁻¹ | {ho_} |")
    A("")
    A("![スポーク](036_fig_spokes.png)\n\n**図 5** シーメンススターの 36 本のスポークの変調と、それ以外の角度成分（雑音・アーティファクト）の比。横軸は半径をスポークの周波数に換算した値。▼が SNR = 3 を下回る点。2.5 µm⁻¹ 付近の落ち込みは、スポークのない同心円の位置です。BLIS-FPM では 3.3–4.8 µm⁻¹ で比が約 10 に保たれ、4 枚を除いて解いた場合（点線）もほぼ同じです。\n")
A("**半分平均の FRC**：\n")
A("| 再構成 | FRC が 1/7 を下回る周波数 | 半ビット基準 |\n|---|---|---|")
for k, v in frc.items():
    A(f"| {k} | {v['limit_1_7']:.2f} µm⁻¹ | {v['limit_halfbit']:.2f} µm⁻¹ |")
A("\n- FRC は同じ装置・同じ前処理で得た 2 組のデータの再現性で、系統誤差（背景の差し引き、ホット画素の補間、較正のずれ）は両方に共通に入ります。中央 512 画素の BLIS-FPM（明視野のみ）は、リング 1 の届く範囲を超えたところで FRC が上がるため 1/7 を下回らず、表の値は意味を持ちません。")
A("- スポーク変調は既知の物体の構造（36 回対称）を直接見るので、雑音や系統誤差では作られにくい指標です。暗視野によって 3.5 µm⁻¹ より上のスポークが現れることは、リング 3 のデータが物体の高周波の情報を運んでいることを示します。一方で未使用のリング 3 の画像の予測が 1 前後にとどまることから、暗視野像の強度（星の中心付近の構造や背景）は順モデルで表しきれておらず、分解能の数値は「スポークの向きと位置が再現される周波数」と読むのが適切です。")
A("")
if os.path.exists(W + "d036/pc/pc_summary.json"):
    exec(open(W + "d036/pc/report_pc_section.py", encoding="utf-8").read())
A("## 6. 003・018 との比較\n")
A("| 項目 | 003 | 018 | 036（明視野のみ） |\n|---|---|---|---|")
A("| FZP Δr / k_c | 100 nm / 5.0 µm⁻¹ | 100 nm / 5.0 µm⁻¹ | 200 nm / 2.5 µm⁻¹ |")
A("| 使った画像 | 44（明視野、2 回） | 44（明視野、1 回） | 9（明視野、6 回平均） |")
A(f"| 未使用画像の misfit、DIP 最適・512（平面波 / FOV） | — | {f3(C18['512']['results']['DIP (optimal it.), plane-wave model']['band_misfit_validation_images'])} / {f3(C18['512']['results']['DIP (optimal it.), FOV model']['band_misfit_validation_images'])} | {f3(pB5('DIP (optimal it.)', 'plane-wave model'))} / {f3(pB5('DIP (optimal it.)', 'FOV model'))} |")
A(f"| 同、EPRY・512 | — | {f3(C18['512']['results']['EPRY, plane-wave model']['band_misfit_validation_images'])} / {f3(C18['512']['results']['EPRY, FOV model']['band_misfit_validation_images'])} | {f3(pB5('EPRY', 'plane-wave model'))} / {f3(pB5('EPRY', 'FOV model'))} |")
A(f"| スポーク SNR > 3（DIP 最適・FOV） | — | {C18['512']['results']['DIP (optimal it.), FOV model']['spoke_snr3_um_inv']:.2f} µm⁻¹（512） | {spoke1024['DIP (optimal it.), FOV model']:.2f} µm⁻¹（1024） |")
A("\n- 036 の misfit が 018 より小さいのは、6 回平均で雑音が小さいことと、画像が 9 枚（未使用 2 枚）と少ないことの両方によるもので、手法の優劣の比較にはなりません（未使用の画像も集合も違う）。")
A("- 036 では FOV モデルの効果が大きく出ます。リング 1 の照明が |k|/k_c ≈ 0.8 と瞳の縁に近く、視野の中の位置で局所的な照明角が変わると瞳から外れる部分（口径食）が生じやすいためと考えられます（018 のリングは瞳の内側に広く分布）。")
A("- 分解能は 018 がチャートの上限（約 5 µm⁻¹、k_c の 1.0 倍）で決まっていたのに対し、036 は k_c = 2.5 µm⁻¹ の 1.3–1.4 倍で、信号対雑音比で決まっています。")
A("")
A("## 7. 注意点・確かめていないこと\n")
A("- p = 0.75 m と CZP の焦点距離は実測値ではありません。倍率 1.32 倍の較正結果と、FOV の強さの 2 つの推定（試料なし画像の境界から c ≈ 1.15、BLIS-FPM の misfit から c ≈ 0.5）が食い違っており、比較はすべて公称の c = 1 です。")
A("- ホット画素 18 % を補間しています。補間した画素は独立な測定ではないので、高周波の雑音の見積もりと FRC を楽観的にしている可能性があります。")
A("- ドリフト補正は明視野の位置で測ったずれの 3 次式です。暗視野の位置での補正は直接は確かめていません。")
A("- DIP 1024（明視野のみ）は検証誤差が 40 / 50 反復で最小になり、その後 100 反復改善しなかったため 170 反復で打ち切りました。")
A("- DIP 1024（暗視野あり、28 枚）は計算時間の都合で 150 反復までとしました。選ばれた反復は平面波・FOV とも 130 で、反復を増やせば改善する可能性があります。")
A("- 4 章の暗視野の結果は、部分コヒーレンスを含まない順モデルでの結果です。5 章の部分コヒーレンスの計算は、次の仮定の上に成り立っています。")
A("  - 照明の角度分布 S は「中心（ガウス）＋すそ（Cauchy 型）」という形を仮定しました。試料なし画像からは中心の幅が決まらず、すその割合 η も 0.1 と 0.3 は未使用画像では区別できません。")
A("  - 暗視野の画像には、モデルで説明できない一様なバックグラウンドがあるため、画像ごとに倍率（DIP・EPRY はオフセットも）を当てはめています。比較用の照明 1 本の計算も同じ扱いですが、暗視野の強度の絶対値は使っていないことになります。")
A("  - リング 2 の像は照明モードのまとめ方で 6–17 % 変わります。44 枚の計算（BLIS-FPM のみ）の数値は不確かさが大きめです。")
A("  - DIP の全視野 1024 画素は FOV モデルだけを計算しました（計算時間のため）。")
A("  - コヒーレントな順モデルでは、FOV の二次位相が周期的な格子の端で折り返し、窓の外側に見かけの直接光ができていました（リング 3 で、窓の端にある暗視野強度の平均の約 1/3）。窓関数でほぼ抑えられていますが、部分コヒーレントの計算では、モデルとデータの両方を窓で重み付けした平均で割っています。")
A("")
A("## 8. 改善案\n")
A("1. **部分コヒーレンス（実施済み、5 章）**：照明の角度広がりを入れると、明視野の当てはめと予測はわずかに良くなりましたが、リング 3 の強度は予測できないままでした。リング 3 の系統誤差（照明角のずれ、暗視野のバックグラウンド、検出器の応答）を切り分けることが次の課題です。S の中心の幅を決めるには、瞳の縁（1.0–1.1 k_c）を細かく通る試料なし画像の測定が有効です。")
A("2. **照明角の自己較正**：リング 2・3 の k をステージ座標の外挿でなく、画像ごとに最適化する（今回の ±3 % の一様な倍率変更では改善せず、個別の誤差や回転の確認が必要）。")
A("3. **暗視野の信号対雑音比**：リング 3 は明視野の約 0.03 % の強度で、6 回平均でも雑音が帯域の半分程度を占めます。暗視野の位置だけ露光を 10 倍以上にするか、照明をリング 1 と 2 の間（|k|/k_c ≈ 1.0–1.2）に増やす方が効率的です。")
A("4. **検出器**：画素の 18 % がホット画素で、暗視野の弱い信号を最も大きく損ないます。冷却・交換、またはホット画素の少ない領域への試料の配置を勧めます。")
A("5. **ドリフト**：6 回の間に 2.7 µm 動いています。試料ステージの温度安定化、または各フレームの位置ずれを再構成の中で推定する方法が有効です。")
A("6. **p の実測**：FOV モデルの効果が大きいデータなので、試料–対物距離（と CZP の集光条件）を測ることが、FOV モデルの信頼性に直結します。")
A("")
A("## ファイル\n")
A("- `pc/`：部分コヒーレンスの計算（5 章）。`pc_summary.json`（数値すべて）、`out_512_pc*`・`out_1024_pc*`（比較図・TIFF）、`blis/`（BLIS-FPM の解の数値）。")
A("- `036_summary.json`：本レポートの数値すべて。`out_512*/`・`out_1024*/`：比較図・数値（`compare_summary.json`）・TIFF。")
A("- `_df`：暗視野を含む 28 枚（帯域制限なしの物体で評価）、`_df_ho`：BLIS-FPM の 4 枚を除いた解、`_df_bp`：同じ 28 枚を帯域制限した物体で評価、`_objband(_ho)`：BLIS-FPM の物体帯域制限版（と 2 枚を除いた解）。")
open(W + "d036/036_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", len(L), "lines")
