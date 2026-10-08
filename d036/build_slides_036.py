# -*- coding: utf-8 -*-
# FPT #036 summary deck (16:9, Yu Gothic). All numbers come from the analysis JSONs via make_report_036.py.
import os
W = "./"
exec(open(W + "build_pptx.py", encoding="utf-8").read())          # helpers: textbox, title, footer, picture, split_comparison, table
OUT = W + "d036/deck/"; os.makedirs(OUT, exist_ok=True)
FOOT = "FPT #036 シーメンススター 再構成まとめ（2026-09-26）"
ns = {"__file__": W + "d036/make_report_036.py"}; exec(open(W + "d036/make_report_036.py", encoding="utf-8").read(), ns)
g = lambda k: ns[k]
C, M, ML, MOD, per_ring, r = g("C"), g("M"), g("ML"), g("MOD"), g("per_ring"), g("r")
prep, cal, sP, sF, bcP, bcF = g("prep"), g("cal"), g("sP"), g("sF"), g("bcP"), g("bcF")
kk, ring, cover, drift, lev, nf512, nf1024 = g("kk"), g("ring"), g("cover"), g("drift"), g("lev"), g("nf512"), g("nf1024")
spoke1024, frcL, kt, b13, b44, ediag, C18, kdir, cbest = g("spoke1024"), g("frcL"), g("kt"), g("b13"), g("b44"), g("ediag"), g("C18"), g("kdir"), g("cbest")
f3 = lambda v: f"{v:.3f}"
bh = C["512_objband_ho"]["results"]
FIG = dict(data="d036/036_fig_data.png", cal="fpt_pipeline/fpt_output_036bf/figures/fig1_calibration.png",
           rec="fpt_pipeline/fpt_output_036bf/figures/fig2_reconstruction.png", fov="d036/036_fig_fov_scan.png",
           c512f="d036/out_512/fig_compare_FOV_c1.png", c512p="d036/out_512/fig_compare_planewave.png",
           c1024f="d036/out_1024/fig_compare_FOV_c1.png", dtrain="dip_pipeline/dip_output_036bf/fig_dip_training.png",
           esweep="epry_pipeline/epry_output_036bf/fig_epry_sweeps.png", df="d036/036_fig_darkfield.png",
           cdf="d036/out_512_df/fig_compare_FOV_c1.png", cdf1024="d036/out_1024_df/fig_compare_FOV_c1.png", star="d036/036_fig_df_starzoom.png")
FIG = {k: W + v for k, v in FIG.items()}

prs = Presentation(); prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
BL = prs.slide_layouts[6]; n = 0
def new():
    global n
    n += 1; s = prs.slides.add_slide(BL)
    if n > 1: footer(s, n)
    return s
pB = lambda t, m, mod, k="band_misfit_validation_images": r(t, m, mod)[k]

# 1 ---- title
s = new()
band = s.shapes.add_shape(1, 0, Inches(2.2), Inches(SW), Inches(2.6)); band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
textbox(s, 0.8, 2.45, 11.8, 1.0, ["X 線フーリエタイコグラフィ #036 シーメンススターの再構成"], size=28, color=WHITE, bullets=False)
textbox(s, 0.8, 3.45, 11.8, 0.6, ["DIP（最適反復・30 反復）・EPRY・BLIS-FPM：明視野のみ → 暗視野を含む場合"], size=22, color=WHITE, bullets=False)
textbox(s, 0.8, 5.2, 11.8, 1.2, ["データ：036（30 keV、FZP Δr = 200 nm、44 照明位置 × 試料あり/なし × 6 回）", "2026-09-26"],
        size=16, color=GREY, bullets=False)
for p in s.shapes[1].text_frame.paragraphs:
    for rr in p.runs: rr.font.bold = True

# 2 ---- summary
s = new(); title(s, "まとめ")
pr = lambda t, m, mod="FOV model": per_ring(t, m, mod)
items = [
    f"**前処理**：6 回の撮影の間に試料が {abs(drift[0]):.1f} / {abs(drift[1]):.1f} µm ずれた → 相互相関で補正して中央値平均。画素の {prep['hot_pixel_fraction'] * 100:.0f} % がホット画素（置換）",
    f"**照明**：リング 1（9 点、|k|/k_c ≈ 0.8）明視野、リング 2（16 点、≈ 1.34）境界、リング 3（19 点、≈ 1.87）暗視野。較正はリング 1 で（misfit {f3(cal['misfit'])}、倍率 {cal['scale_vs_nominal']:.2f} 倍、符号 −1）",
    f"**明視野のみ（9 枚）**：全手法で **FOV モデルが明確に良い**（未使用画像、512 画素：DIP 最適 {f3(pB('512', 'DIP (optimal it.)', 'plane-wave model'))} → {f3(pB('512', 'DIP (optimal it.)', 'FOV model'))}、EPRY {f3(pB('512', 'EPRY', 'plane-wave model'))} → {f3(pB('512', 'EPRY', 'FOV model'))}、BLIS-FPM {f3(bh['BLIS-FPM, plane-wave model']['band_misfit_validation_images'])} → {f3(bh['BLIS-FPM, FOV model']['band_misfit_validation_images'])}）。最良は **DIP 最適・FOV**（512：{f3(pB('512', 'DIP (optimal it.)', 'FOV model'))}、1024：{f3(pB('1024', 'DIP (optimal it.)', 'FOV model'))}）",
    f"**分解能**（1024、スポーク SNR > 3）：4 手法とも {min(spoke1024.values()):.2f}–{max(spoke1024.values()):.2f} µm⁻¹ ＝ k_c の 1.3–1.4 倍。信号対雑音比（S = N ≈ 3.0–3.4 µm⁻¹）で決まる",
    f"**暗視野**：3 パイプラインに暗視野処理を追加（既定は不変）。リング 2 は直接光の裾と縁の干渉縞が支配 → 除外。リング 1 + 3 の 28 枚では、**使っていないリング 3 の画像を 4 手法とも予測できない**（misfit：DIP 最適 {f3(pr('512_df', 'DIP (optimal it.)')['ho3'])}、EPRY {f3(pr('512_df', 'EPRY')['ho3'])}、BLIS-FPM {f3(pr('512_df_ho', 'BLIS-FPM')['ho3'])}；1 = 予測なし）",
    f"**超解像**（1024、スポーク SNR > 3、FOV）：暗視野を加えると **BLIS-FPM {spoke1024['BLIS-FPM, FOV model']:.2f} → {r('1024_df', 'BLIS-FPM', 'FOV model')['spoke_snr3_um_inv']:.2f} µm⁻¹**（4 枚除外でも {r('1024_df_ho', 'BLIS-FPM', 'FOV model')['spoke_snr3_um_inv']:.2f}、半周期 {1e3 / (2 * r('1024_df', 'BLIS-FPM', 'FOV model')['spoke_snr3_um_inv']):.0f} nm ≈ 1.9 k_c）、DIP 最適 {spoke1024['DIP (optimal it.), FOV model']:.2f} → {r('1024_df', 'DIP (optimal it.)', 'FOV model')['spoke_snr3_um_inv']:.2f}、EPRY {spoke1024['EPRY, FOV model']:.2f} → {r('1024_df', 'EPRY', 'FOV model')['spoke_snr3_um_inv']:.2f}。暗視野は細いスポークの情報を運ぶが、暗視野像の強度は予測できず、高周波の振幅・細部は未検証",
]
if os.path.exists(W + "d036/pc/pc_summary.json"):
    import json as _js
    _TQ = _js.load(open(W + "d036/pc/pc_summary.json"))["tables"]; _gq = lambda t, m, k="ho1": _TQ[t]["results"][f"{m}, FOV model"][k]
    _iq = [100 * (1 - _gq(t + "_pc", m) / _gq(t + "_pc1m", m)) for t in ("512", "1024") for m in M]
    _h3 = [_gq(t + "_pc", m, "ho3") for t in ("512", "1024") for m in M]
    items.append(f"**部分コヒーレンス**：照明の角度広がり（直接光のすそ）を約 20 本の平面波の和で表すと、28 枚の計算で未使用の明視野画像が 4 手法とも {min(_iq):.0f}–{max(_iq):.0f} % 改善（明視野だけの 9 枚では改善なし）。"
                 f"リング 3 は {min(_h3):.2f}–{max(_h3):.2f} で予測できないまま → 原因は角度広がりではない")
textbox(s, 0.6, 1.15, 12.1, 5.9, items, size=14, space=6)

# 3 ---- data
s = new(); title(s, "測定データと前処理")
picture(s, FIG["data"], 0.4, 1.12, w=7.9)
textbox(s, 8.45, 1.1, 4.5, 5.9, [(("h"), "測定"),
    "30 keV、実効画素 31.9 nm、1024²",
    "FZP Δr = 200 nm、φ155 µm → k_c = 2.5 µm⁻¹",
    "44 照明 × 試料あり/なし × 6 回（各 1 s）",
    ("h", "前処理"),
    f"背景：暗視野の試料なし画像の中央値（{prep['background_median']:.0f} counts）を差し引く",
    f"ホット画素 {prep['hot_pixel_fraction'] * 100:.1f} % を 5×5 近傍の平均で置換",
    f"ドリフト：明視野比画像の相互相関 → 3 次式（残差 {prep['drift_fit_residual_rms_px'][0]:.2f} / {prep['drift_fit_residual_rms_px'][1]:.2f} px）。試料ありフレームだけ戻す",
    "6 回の中央値平均。奇数回・偶数回の半分平均も作成（雑音・FRC 用）",
    f"暗視野：S − D を明視野レベルで規格化。リング 2 ≈ {lev[2] * 100:.2f} %、リング 3 ≈ {lev[3] * 100:.3f} %"], size=12.5, space=4)

# 4 ---- calibration & classes
s = new(); title(s, "較正・照明の分類・視野効果の強さ")
picture(s, FIG["cal"], 0.4, 1.15, w=6.6)
picture(s, FIG["fov"], 7.35, 3.75, w=3.9)
textbox(s, 7.2, 1.1, 5.7, 2.6, [
    f"WOTF 較正（リング 1）：回転 {cal['rotation_deg']:+.2f}°、倍率 {cal['scale_vs_nominal']:.3f} 倍（{cal['angle_per_pulse_urad']:.3f} µrad/pulse）、デフォーカス {cal['defocus_um']:.0f} µm、misfit {f3(cal['misfit'])}",
    f"位相の符号：corr(a, φ) = {cal['corr_a_phi_linear']:+.2f} → −1（018 と同じ）",
    f"リング 2：視野の端 {cover[ring == 2].min() * 100:.0f}–{cover[ring == 2].max() * 100:.0f} % に直接光。リング 3：なし",
    f"FOV の強さ c：試料なし画像の境界から {kdir['c']:.2f}、BLIS-FPM の misfit 走査では ≈ {cbest[0]:.1f}（右下）→ 比較は公称 c = 1"], size=12.5, space=5)

# 5 ---- BLIS bright field
s = new(); title(s, "BLIS-FPM：明視野 9 枚・全視野")
picture(s, FIG["rec"], 0.4, 1.15, w=7.7, h=5.8)
textbox(s, 8.3, 1.2, 4.7, 5.6, [
    f"misfit：平面波 {f3(sP['final_misfit'])} → FOV {f3(sF['final_misfit'])}（{(1 - sF['final_misfit'] / sP['final_misfit']) * 100:.0f} % 減）",
    f"デフォーカス：{sP['defocus_equiv_um']:.0f} / {sF['defocus_equiv_um']:.0f} µm",
    f"FOV 解の位相パワーの {bcF['phase_power_fraction']['band'] * 100:.1f} % は 0.3–3.5 µm⁻¹ → 帯域制限しても {f3(bcF['object 0.3-3.5']['misfit'])}（平面波 {f3(bcP['object 0.3-3.5']['misfit'])}）",
    "**018 と違い、FOV の改善は帯域外への逃げではない**",
    f"2 枚（画像 0・5）を除いて解くと、その 2 枚で 平面波 {f3(bh['BLIS-FPM, plane-wave model']['band_misfit_validation_images'])}、FOV {f3(bh['BLIS-FPM, FOV model']['band_misfit_validation_images'])}（DIP 最適 {f3(pB('512', 'DIP (optimal it.)', 'FOV model'))} より悪い）"],
    size=13, space=6)

# 6 ---- 512 comparison
s = new(); title(s, "明視野のみ・中央 512 画素：4 手法の比較（FOV モデル）")
top, bot, _, _ = split_comparison(FIG["c512f"], "c036_512f")
picture(s, top, 0.35, 1.1, w=5.0, h=5.9)
rows = [["手法", "平面波 学習", "平面波 未使用", "FOV 学習", "FOV 未使用"]]
for m in M:
    a, b = r("512", m, "plane-wave model"), r("512", m, "FOV model")
    st = "＊" if m == "BLIS-FPM" else ""
    rows.append([ML[m], f3(a["band_misfit_training_images"]), f3(a["band_misfit_validation_images"]) + st, f3(b["band_misfit_training_images"]), f3(b["band_misfit_validation_images"]) + st])
rows.append(["BLIS-FPM（2 枚除外）", f3(bh["BLIS-FPM, plane-wave model"]["band_misfit_training_images"]), f3(bh["BLIS-FPM, plane-wave model"]["band_misfit_validation_images"]),
             f3(bh["BLIS-FPM, FOV model"]["band_misfit_training_images"]), f3(bh["BLIS-FPM, FOV model"]["band_misfit_validation_images"])])
table(s, rows, 5.55, 1.2, 7.4, col_w=[2.4, 1.2, 1.3, 1.2, 1.3], size=11, row_h=0.36)
textbox(s, 5.55, 3.9, 7.4, 3.0, ["misfit：強度コントラストの 0.3–3.5 µm⁻¹ の相対誤差（未使用 = 画像 0・5）。＊ 9 枚すべてで解いた値",
    f"DIP の選択反復：{r('512', 'DIP (optimal it.)', 'plane-wave model')['iteration']}（平面波）/ {r('512', 'DIP (optimal it.)', 'FOV model')['iteration']}（FOV）",
    "4 手法の位相は 1–3 µm⁻¹ でほぼ一致（FRC > 0.95）。差は 3 µm⁻¹ 以上と低周波",
    "平面波モデルの像：out_512/fig_compare_planewave.png"], size=12.5, space=5)

# 7 ---- 1024 comparison
s = new(); title(s, "明視野のみ・全視野 1024 画素（FOV モデル）とスポーク分解能")
top, bot, _, _ = split_comparison(FIG["c1024f"], "c036_1024f")
picture(s, top, 0.35, 1.1, w=5.0, h=5.9)
rows = [["手法", "平面波 未使用", "FOV 未使用", "スポーク 平面波", "スポーク FOV"]]
for m in M:
    st = "＊" if m == "BLIS-FPM" else ""
    rows.append([ML[m], f3(pB("1024", m, "plane-wave model")) + st, f3(pB("1024", m, "FOV model")) + st,
                 f"{spoke1024[m + ', plane-wave model']:.2f}", f"{spoke1024[m + ', FOV model']:.2f}"])
table(s, rows, 5.55, 1.2, 7.4, col_w=[2.2, 1.3, 1.3, 1.3, 1.3], size=11, row_h=0.36)
textbox(s, 5.55, 3.2, 7.4, 3.6, ["スポーク：36 本の変調の SNR > 3 が続く最低周波数（µm⁻¹）、星の中心 (288, 525) 画素。＊ BLIS-FPM は 9 枚すべてで解いた値",
    f"4 手法とも {min(spoke1024.values()):.2f}–{max(spoke1024.values()):.2f} µm⁻¹（半周期 {1e3 / (2 * max(spoke1024.values())):.0f}–{1e3 / (2 * min(spoke1024.values())):.0f} nm）＝ k_c の 1.3–1.4 倍",
    "中央 512 画素は星の中心を含まないのでスポーク解析は不可",
    "DIP 1024 は検証誤差が 40 / 50 反復で最小、170 反復で打ち切り"], size=12.5, space=5)

# 8 ---- dark field: method
s = new(); title(s, "暗視野への拡張：データの作り方とパイプラインの変更")
textbox(s, 0.6, 1.15, 12.1, 5.9, [
    ("h", "データ"),
    "暗視野・境界の画像は試料なし画像で割れない → 各フレームで [ S − G₃{D} ] / ( g · U )、ドリフト補正して中央値平均",
    "  G₃{D}：同じ位置の試料なし画像（直接光の裾・迷光を差し引く）、g：その回の明視野レベル、U：照明の包絡",
    f"雑音の割合（0.3–3.5 µm⁻¹、512 画素）：リング 1 / 2 / 3 = {nf512[1]:.3f} / {nf512[2]:.2f} / {nf512[3]:.2f} → リング 3 は信号 ≈ 雑音",
    ("h", "順モデルと各手法の変更（設定で切り替え、003/018 の既定動作は不変）"),
    "|k| > 1.02 k_c の画像は直接光で規格化しない（明視野 = 1 の共通スケール）。試料なしが明るい画素は除外",
    "DIP：暗視野画像に雑音の逆数の重み、強度係数は最小二乗で毎回決定（学習パラメータのままだと 1.01 から動かず失敗）",
    f"EPRY：強度係数を毎掃引推定すると 0 に縮む（リング 3 で {ediag['collapse']['hist'][0]['g_ring3']:.2f} → {ediag['collapse']['hist'][-1]['g_ring3']:.3f}）→ 最初の更新時に 1 回推定して固定",
    "BLIS-FPM：画像ごとの雑音の逆数の重み（コントラスト単位）と有効画素マスク",
    ("h", "リング 2 を除外した理由"),
    f"44 枚で BLIS-FPM を解いてもリング 2 の主な構造（縁の明るい部分・同心円の縞）はモデルに現れない（misfit {b44['s-1']['mean_per_ring']['2']:.2f}）。照明の角度広がりで直接光の裾が瞳に入るため → 28 枚（リング 1 + 3）で比較"],
    size=13, space=4)


def df_images_slide(tag, key, ttl, extra):
    """Dark-field reconstruction images (FOV model): image grid (left), FRC / pupil / misfit row and notes (right)."""
    s = new(); title(s, ttl)
    top, bot, _, _ = split_comparison(FIG[key], "c036_" + tag)
    picture(s, top, 0.3, 1.1, w=5.65, h=5.9)
    from PIL import Image as _I
    wb, hb = _I.open(bot).size; hh = 6.95 * hb / wb
    picture(s, bot, 6.1, 1.15, w=6.95)
    g = C[tag]["grey_scale_FOV_c1"]
    rr = lambda m: r(tag, m, "FOV model")
    lines = [f"表示は明視野のスライドと同じ：0.3–3.5 µm⁻¹ に帯域制限、濃淡の範囲は 4 手法共通（透過率 {g['transmission'][0]:.3f}–{g['transmission'][1]:.3f}、位相 {g['phase_rad'][0]:.3f}–{g['phase_rad'][1]:.3f} rad）",
             f"学習 {C[tag]['images_train']} 枚（リング 1 の 8 枚 + リング 3 の 16 枚）、未使用 {C[tag]['images_validation']} 枚（画像 0・25・34・43）。BLIS-FPM は 28 枚すべてで解いた全視野の解",
             "吸収 a の帯域内 rms：" + "、".join(f"{ML[m]} {rr(m)['absorption_rms_band']:.4f}" for m in M) + "（EPRY と BLIS-FPM の透過率は起伏が大きい）"] + extra + [
             f"右上：位相の FRC（DIP 最適反復との比較）、瞳の Zernike 係数、misfit。平面波モデルの像は Drive の figures/df_{tag.split('_')[0]}_compare_planewave.png"]
    textbox(s, 6.1, 1.15 + hh + 0.12, 6.95, 7.0 - (1.15 + hh + 0.12), lines, size=11, space=3)
    return s

# 9a ---- dark field: reconstruction images, 512
_e0 = r('512_df', 'EPRY', 'FOV model'); _eb = r('512', 'EPRY', 'FOV model')
df_images_slide('512_df', 'cdf', '暗視野を含む 28 枚・中央 512 画素：再構成像（FOV モデル）', [
    f"DIP（最適反復）の選択反復：{r('512_df', 'DIP (optimal it.)', 'FOV model')['iteration']}",
    f"EPRY は未使用画像の misfit が {_e0['iteration'] + 1} 回目の掃引で最小になり、その像を表示（位相の帯域内 rms：明視野のみ {_eb['phase_rms_band']:.3f} → {_e0['phase_rms_band']:.3f} rad）"])

# 9 ---- dark field: results
s = new(); title(s, "暗視野を含む 28 枚・中央 512 画素：未使用の暗視野画像は予測できない")
picture(s, FIG["df"], 0.35, 1.1, w=7.7)
rows = [["手法（FOV）", "R1 学習", "R1 未使用", "R3 学習", "R3 未使用"]]
for m in M:
    v = per_ring("512_df_ho" if m == "BLIS-FPM" else "512_df", m, "FOV model")
    rows.append([ML[m], f3(v["fit1"]), f3(v["ho1"]), f3(v["fit3"]), f3(v["ho3"])])
table(s, rows, 8.25, 1.2, 4.75, col_w=[1.9, 0.95, 1.05, 0.95, 1.05], size=10.5, row_h=0.34)
textbox(s, 8.25, 3.05, 4.75, 3.9, ["中央 512 画素、帯域制限しない物体で評価。未使用：画像 0（R1）、25・34・43（R3）。1.0 = 予測なし",
    f"学習に使ったリング 3 によく合うのは BLIS-FPM だけ（雑音の割合を下回り、雑音まで当てはめ）。DIP・EPRY は学習画像でも {min(pr('512_df', m)['fit3'] for m in M if m != 'BLIS-FPM'):.2f}–{max(pr('512_df', m)['fit3'] for m in M if m != 'BLIS-FPM'):.2f}。未使用のリング 3 は全手法 ≳ 1",
    "BLIS-FPM で R3 の k を ±3 %、瞳半径を ±4 % 変えても未使用 R3 は 1.1 以上",
    f"半分平均の FRC は延びる（EPRY {frcL['EPRY, bright field (ring 1), centre 512']:.1f} → {frcL['EPRY, bright + dark field (rings 1, 3), centre 512']:.1f}、BLIS {frcL['BLIS-FPM, bright field (ring 1), full field']:.1f} → {frcL['BLIS-FPM, bright + dark field (rings 1, 3), full field']:.1f} µm⁻¹）。共通の系統誤差も再現されるので、次のスポーク解析と合わせて判断"],
    size=11.5, space=5)
if "1024_df" in C:
    _e1 = r('1024_df', 'EPRY', 'FOV model')
    df_images_slide('1024_df', 'cdf1024', '暗視野を含む 28 枚・全視野 1024 画素：再構成像（FOV モデル）', [
        f"DIP（最適反復）の選択反復：{r('1024_df', 'DIP (optimal it.)', 'FOV model')['iteration']}（150 反復で打ち切り）",
        f"EPRY は {_e1['iteration'] + 1} 回目の掃引の像（未使用画像の misfit が最小）",
        f"BLIS-FPM の透過率は視野の周辺で起伏が大きい（吸収 a の rms：明視野のみ {r('1024', 'BLIS-FPM', 'FOV model')['absorption_rms_band']:.4f}、暗視野あり {r('1024_df', 'BLIS-FPM', 'FOV model')['absorption_rms_band']:.4f}）"])
    s = new(); title(s, "暗視野を含む 28 枚・全視野 1024 画素：スポーク分解能は上がる")
    picture(s, W + "d036/036_fig_spokes.png", 0.35, 1.2, w=8.0)
    rows = [["手法（FOV）", "R1 未使用", "R3 学習", "R3 未使用", "スポーク 明視野のみ", "スポーク 暗視野あり"]]
    for m in M:
        tt = "1024_df_ho" if (m == "BLIS-FPM" and "1024_df_ho" in C) else "1024_df"
        v = per_ring(tt, m, "FOV model")
        rows.append([ML[m], f3(v["ho1"]), f3(v["fit3"]), f3(v["ho3"]), f"{spoke1024[m + ', FOV model']:.2f}", f"{r('1024_df', m, 'FOV model')['spoke_snr3_um_inv']:.2f}"])
    table(s, rows, 0.45, 4.7, 8.0, col_w=[2.0, 1.0, 1.0, 1.0, 1.3, 1.3], size=10.5, row_h=0.34)
    textbox(s, 8.65, 1.2, 4.35, 5.6, ["図：36 本のスポークの変調 / それ以外の角度成分。▼ が SNR 3 を下回る点",
        f"BLIS-FPM は 3.3–4.8 µm⁻¹ で比 ≈ 10 を保つ（4 枚除外の解も同じ）→ **リング 3 は星の高周波の情報を運んでいる**",
        f"一方、未使用のリング 3 の強度はほとんど予測できない（DIP 最適 {f3(per_ring('1024_df', 'DIP (optimal it.)', 'FOV model')['ho3'])}、DIP 30 反復 {f3(per_ring('1024_df', 'DIP (30 it.)', 'FOV model')['ho3'])}、EPRY {f3(per_ring('1024_df', 'EPRY', 'FOV model')['ho3'])}、BLIS-FPM {f3(per_ring('1024_df_ho', 'BLIS-FPM', 'FOV model')['ho3'])}）→ 暗視野像の強度は順モデルで表しきれていない",
        "分解能の数値は「スポークの向きと位置が再現される周波数」と読むのが適切",
        "DIP 1024（暗視野あり）は 150 反復まで（選択 130）"], size=12, space=6)
    s = new(); title(s, "星の中心の位相：暗視野でスポークの届く半径が内側へ")
    picture(s, FIG["star"], 0.35, 1.15, w=8.7)
    _sp = lambda t, m: r(t, m, "FOV model")["spoke_snr3_um_inv"]
    textbox(s, 9.3, 1.2, 3.75, 5.7, ["星の中心 (288, 525) 画素のまわり 160 画素（5.1 µm）角。帯域制限しない位相（FOV モデル）、濃淡は各パネルの 1–99 %",
        "上段：明視野のみ（リング 1、9 枚）。下段：明視野 + 暗視野（リング 1 + 3、28 枚）",
        "円：36 本のスポークの変調が SNR 3 を下回る半径。数値はその周波数で、円が小さいほど細いスポークまで再現",
        "スポーク SNR > 3 の周波数、明視野のみ → 暗視野あり（µm⁻¹）：" + "、".join(f"{ML[m]} {_sp('1024', m):.2f} → {_sp('1024_df', m):.2f}" for m in M),
        "BLIS-FPM は最も内側まで届くが、細かい起伏（雑音を含む）も像全体に増える。暗視野像の強度は予測できていないので、細部の振幅までは確かめられていない"],
        size=11.5, space=5)

# 10pc ---- partial coherence (4 slides)
if os.path.exists(W + "d036/pc/pc_summary.json"):
    exec(open(W + "d036/pc/slides_pc.py", encoding="utf-8").read())

# 10 ---- comparison with 003 / 018
s = new(); title(s, "003・018 との比較")
R18 = C18["512"]["results"]
rows = [["項目", "018", "036（明視野のみ）"],
        ["FZP Δr / k_c", "100 nm / 5.0 µm⁻¹", "200 nm / 2.5 µm⁻¹"],
        ["使った画像", "44（明視野、1 回撮影）", "9（明視野、6 回平均）"],
        ["未使用画像 DIP 最適・512（平面波 / FOV）", f"{f3(R18['DIP (optimal it.), plane-wave model']['band_misfit_validation_images'])} / {f3(R18['DIP (optimal it.), FOV model']['band_misfit_validation_images'])}",
         f"{f3(pB('512', 'DIP (optimal it.)', 'plane-wave model'))} / {f3(pB('512', 'DIP (optimal it.)', 'FOV model'))}"],
        ["未使用画像 EPRY・512（平面波 / FOV）", f"{f3(R18['EPRY, plane-wave model']['band_misfit_validation_images'])} / {f3(R18['EPRY, FOV model']['band_misfit_validation_images'])}",
         f"{f3(pB('512', 'EPRY', 'plane-wave model'))} / {f3(pB('512', 'EPRY', 'FOV model'))}"],
        ["スポーク SNR > 3（DIP 最適・FOV）", f"{R18['DIP (optimal it.), FOV model']['spoke_snr3_um_inv']:.2f} µm⁻¹（チャートの上限）", f"{spoke1024['DIP (optimal it.), FOV model']:.2f} µm⁻¹（1.4 k_c、SNR で決まる）"]]
table(s, rows, 0.6, 1.25, 12.1, col_w=[4.2, 3.9, 4.0], size=12, row_h=0.42)
textbox(s, 0.6, 4.1, 12.1, 2.8, [
    "036 の misfit が小さいのは 6 回平均で雑音が小さく、画像も 9 枚と少ないため。手法の優劣の比較にはならない",
    "036 では FOV モデルの効果が大きい：リング 1 が |k|/k_c ≈ 0.8 と瞳の縁に近く、視野内で局所的な照明角が変わると口径食が起きやすい",
    "018・036 とも、明視野の未使用画像への予測は DIP（最適反復）が最良。BLIS-FPM は学習画像への当てはめすぎの傾向"], size=13.5, space=6)

# 11 ---- caveats & next steps
s = new(); title(s, "注意点と改善案")
textbox(s, 0.6, 1.15, 5.9, 5.8, [("h", "注意点"),
    "p = 0.75 m・CZP 焦点距離は実測でない。FOV の強さの 2 つの推定（1.15 と 0.5）が不一致",
    f"ホット画素 {prep['hot_pixel_fraction'] * 100:.0f} % を補間 → 高周波の雑音・FRC を楽観的にしうる",
    "ドリフト補正は明視野位置で測った 3 次式。暗視野位置では直接確認していない",
    "部分コヒーレンスの角度分布 S は仮定した形（中心＋すそ）。中心の幅は決めきれず、リング 2 の像はモードの表し方に敏感"], size=13, space=6)
textbox(s, 6.8, 1.15, 6.0, 5.8, [("h", "改善案"),
    "部分コヒーレンスは実施済み（明視野がわずかに改善、リング 3 は予測できないまま）→ 次はリング 3 の系統誤差（照明角・バックグラウンド・検出器）の切り分け",
    "リング 2・3 の照明角を画像ごとに自己較正",
    "暗視野位置の露光を 10 倍以上に、または |k|/k_c ≈ 1.0–1.2 に照明を追加",
    "検出器のホット画素対策、試料ステージのドリフト対策（または再構成内での位置推定）",
    "p（試料–対物距離）と CZP の集光条件の実測"], size=13, space=6)

# 12 ---- files
s = new(); title(s, "ファイル構成（Google Drive: FPT036_成果物まとめ_20260926）")
textbox(s, 0.6, 1.2, 12.1, 5.7, [
    "036_report.md、FPT036_再構成まとめ.pptx、036_summary.json（全数値）、figures/（図）",
    "01_データと前処理：前処理・ドリフト・雑音・照明分類・較正の JSON",
    "02_明視野_512 / 03_明視野_1024：DIP（最適・30 反復）・EPRY の TIFF と比較の数値",
    "04_BLIS-FPM_明視野：全視野 9 枚の TIFF・瞳・帯域チェック・視野効果の走査・物体帯域制限の検証",
    "05_暗視野_512 / 06_暗視野_1024：28 枚（リング 1 + 3）の 4 手法の TIFF と比較の数値",
    "07_暗視野_診断：BLIS-FPM 44 枚・28 枚・k/瞳テスト、EPRY の方式比較、半分平均の FRC",
    "08_パイプライン：変更後の dipfpm / eprfpm / fptrecon（036 用設定、暗視野・部分コヒーレンス対応を含む）と解析スクリプトの zip",
    "09_部分コヒーレンス：直接光・S の推定・予測テスト・η の選択・比較の数値、1024 画素の TIFF（部分コヒーレント / 照明 1 本）",
    "TIFF：float32、31.9 nm/px、物理フレーム（位相の符号 −1）。*_unfiltered.tif は帯域制限なし"], size=13, space=6)

path = OUT + "FPT036_再構成まとめ.pptx"; prs.save(path); print("saved", path, n, "slides")
