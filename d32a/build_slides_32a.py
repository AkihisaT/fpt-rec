# -*- coding: utf-8 -*-
"""FPT 32a summary deck (16:9, Yu Gothic). All numbers come from the analysis JSONs via make_report_32a.py.
usage (workspace root): python d32a/build_slides_32a.py  -> d32a/deck/FPT32a_再構成まとめ.pptx"""
import os, json
import numpy as np
W = "./"
FOOT = "FPT 32a 平行ビーム照明・対物 FZP 走査 シーメンススター 再構成まとめ（2026-09-27）"
exec(open(W + "d32a/deck_helpers.py", encoding="utf-8").read())          # textbox, title, footer, picture, table
OUT = W + "d32a/deck/"; os.makedirs(OUT, exist_ok=True)
ns = {"__file__": os.path.abspath(W + "d32a/make_report_32a.py")}; exec(open(W + "d32a/make_report_32a.py", encoding="utf-8").read(), ns)
g = ns.get
C, M, ML, MOD, rr, f3, f2, best, spk = g("C"), g("M"), g("ML"), g("MOD"), g("r"), g("f3"), g("f2"), g("best"), g("spk")
MAIN, OLD, PAR, AFF, ANI, AST, cdf = g("MAIN"), g("OLD"), g("PAR"), g("AFF"), g("ANI"), g("AST"), g("cdf")
AFF2, PAR1, PREV, RES2, PBT = g("AFF2"), g("PAR1"), g("PREV"), g("RES2"), g("PBT")
if AFF2: FOOT = FOOT.replace("2026-09-27", "2026-09-28")
geo, rb, rm_rms, fd, dr, fs, calb, kr = g("geo"), g("rb"), g("rm_rms"), g("fd"), g("dr"), g("fs"), g("calb"), g("kr")
lev6, lev7, nz6, nz7, c_eff, svA, dke = g("lev6"), g("lev7"), g("nz6"), g("nz7"), g("c_eff"), g("svA"), g("dke")
FIG = lambda k, f: W + f"d32a/out_{k}/{f}"
PL, FV = "plane-wave model", "FOV model"

prs = Presentation(); prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
BL = prs.slide_layouts[6]; n = 0
def new():
    global n
    n += 1; s = prs.slides.add_slide(BL)
    if n > 1: footer(s, n)
    return s
def ho(key, m, mod, grp="all"):
    x = rr(key, m, mod)
    return None if x is None else x["misfit_unfiltered"]["heldout"].get(grp)
def it(key, m, mod):
    x = rr(key, m, mod); v = None if x is None else x["iteration"]
    return "—" if v is None else str(v)
def sp(key, m, mod):
    x = rr(key, m, mod); return f2(None if x is None else x.get("spoke_snr3_um_inv"))

# 1 ---- title
s = new()
band = s.shapes.add_shape(1, 0, Inches(2.2), Inches(SW), Inches(2.6)); band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
textbox(s, 0.8, 2.45, 11.8, 1.0, ["X 線フーリエタイコグラフィ 32a：平行ビーム照明・対物 FZP 走査"], size=26, color=WHITE, bullets=False)
textbox(s, 0.8, 3.45, 11.8, 0.9, ["シーメンススターの再構成　DIP（最適反復・30 反復）・EPRY・BLIS-FPM × 平面波 / FOV モデル"], size=20, color=WHITE, bullets=False)
textbox(s, 0.8, 5.2, 11.8, 1.2, ["データ：32a（30 keV、260 位置 × 試料あり/なし）",
                                 "再構成像はすべて帯域フィルターなし　／　" + ("2026-09-28（改訂：物体側のピントずれを瞳に移して解き直し）" if AFF2 else "2026-09-27")], size=16, color=GREY, bullets=False)
for p in s.shapes[1].text_frame.paragraphs:
    for q in p.runs: q.font.bold = True

# 2 ---- summary
s = new(); title(s, "まとめ" + ("（改訂版：物体側のピントずれを瞳に移して解き直し）" if AFF2 else "（改訂版：機械的な像ずれを補正して解き直し）" if AFF else ""))
b512 = best(MAIN["bf512"]); items = [
    f"**条件の修正（ご確認済み）**：対物 FZP は Δr = 250 nm 相当（kc = 2.0 µm⁻¹）、FZP Y は 0.2 µm/単位。Δr = 250 nm の方が外した画像の予測がよい（BLIS-FPM・FOV：{f3(dr['reg2_dr250_c1']['heldout'])} 対 {f3(dr['reg2_dr200_c1']['heldout'])}）"]
if PAR:
    mm = PAR["mechanical"]
    items.append(f"**位置の 2×2 行列（posaffine）**：半パルスの食い違い（39.1 px）を直したあと、像のずれ = 比例（2×2）+ 高次 + ランダムに分解。瞳が予測する分（{100*PAR['fraction_of_drift_explained_by_pupil']:.0f} %）だけ残し、残りを機械誤差として補正。"
                 f"機械的な比例ずれ：列←X {mm['scale_col_pct']:+.2f} %、行←Y {mm['scale_row_pct']:+.2f} %、列←Y {mm['col_from_row_axis_pct']:+.2f} %（回転 {mm['rotation_deg']:+.2f}°）")
    ck = PAR["check"]
    if AFF2:
        L0 = PAR["focus_loop"]
        items.append(f"**楕円状の癖の原因**：以前の版の瞳の大きな非点は、この比例ずれの肩代わり。瞳と物体の 2 次位相は入れ替え可能 → 物体のピントを合わせた和で決定（非点はほぼ 0）")
        rf_ = lambda key, m: rr(key, m, FV)["pupil_drift_check"]["object_refocus"]["z_um"] / 1e3
        items.append(f"**物体側のピントずれを瞳へ（フォーカスループ）**：瞳を固定して物体を解くと物体に {L0[0]['object_refocus']['z_um']/1e3:+.2f} mm のずれ → 瞳へ移して収束：瞳 z {PAR['pupil_defocus_um']/1e3:+.2f} mm。"
                     f"EPRY・BLIS-FPM の物体の再合焦量 {rf_(PREV['bf512'], 'EPRY'):+.2f} / {rf_(PREV['bf512'], 'BLIS'):+.2f} → {rf_(MAIN['bf512'], 'EPRY'):+.2f} / {rf_(MAIN['bf512'], 'BLIS'):+.2f} mm、"
                     f"スポーク {sp(PREV['bf512'], 'EPRY', FV)} / {sp(PREV['bf512'], 'BLIS', FV)} → {sp(MAIN['bf512'], 'EPRY', FV)} / {sp(MAIN['bf512'], 'BLIS', FV)} µm⁻¹（misfit はほぼ不変。縮退のため割り振りは「試料にピントが合う」条件で決定）")
    else:
        items.append(f"**楕円状の癖の原因**：以前の版の瞳の大きな非点は、この比例ずれの肩代わり。瞳と物体の 2 次位相は入れ替え可能 → 物体のピントを合わせた和で決定：z {PAR['pupil_defocus_um']/1e3:+.2f} mm、非点 ({PAR['pupil_astig_um'][0]/1e3:+.2f}, {PAR['pupil_astig_um'][1]/1e3:+.2f}) mm ≈ 0。"
                     f"瞳の予測と残したずれの差 {ck['total_focus_drift_minus_retained_pct']:.3f} %（基準 0.1 %）で整合")
items.append(f"**明視野・中央 512 px**：最良は {b512[1]}・{b512[2]}（外した 7 枚の misfit {f3(b512[0])}）。スポーク分解能 {min(spk(MAIN['bf512'])):.2f}–{max(spk(MAIN['bf512'])):.2f} µm⁻¹")
if MAIN["bf1000"] in C:
    b1 = best(MAIN["bf1000"])
    items.append(f"**明視野・全視野 1000 px**：最良は {b1[1]}・{b1[2]}（{f3(b1[0])}）。スポーク {min(spk(MAIN['bf1000'])):.2f}–{max(spk(MAIN['bf1000'])):.2f} µm⁻¹")
if MAIN["df512"] in C:
    xd = [ho(MAIN["df512"], m, mlab, "dark") for _, mlab, _ in MOD for m, _ in M if rr(MAIN["df512"], m, mlab)]
    items.append(f"**暗視野（リング 6–7、10 s）を追加**：強度は明視野の {100*lev7:.2f}–{100*lev6:.2f} %。外した暗視野像の misfit {min(xd):.2f}–{max(xd):.2f}。暗視野像の位置は、明視野の機械的な 2×2 行列を外挿して補正")
textbox(s, 0.6, 1.15, 12.1, 5.9, items, size=12 if AFF2 else 13, space=5 if AFF2 else 6)

# 3 ---- conditions
s = new(); title(s, "測定条件：ご指定の値と、データから決めた値")
lam = 1.239842 / 30 * 1e-3
rows = [["項目", "ご指定", "使った値", "根拠"],
        ["X 線エネルギー", "30.0 keV", "同じ（λ = 0.041328 nm）", "—"],
        ["実効画素", "31.9 nm/px", "同じ", "—"],
        ["対物 FZP Δr", "200 nm", "**250 nm（kc = 2.0 µm⁻¹）**", "像スペクトルの外縁・ケラレの位置・外した画像の予測。フォルダ名も fzp250nm"],
        ["FZP 直径", "φ155 µm", f"実効 φ{2*geo['NA_edge_lens_shift_um']:.0f} µm", f"ケラレ開始の FZP 移動量 {geo['NA_edge_lens_shift_um']:.1f} µm（Δr 250 nm・f 0.75 m の径 {lam*0.75e6/0.25:.0f} µm）"],
        ["FZP Y の単位", "0.6 µm", "**0.2 µm**", f"検出器/FZP の比 X : Y = 12.5 = 2.5/0.2。リングが円になる（写像の特異値 {svA[0]:.3f} / {svA[1]:.3f}）"],
        ["p（試料–対物）", "0.75 m（未実測）", "FOV モデルの κ = 1/(λp)", f"照明波数はケラレから独立に較正（公称の {calb['scale_vs_nominal']:.3f} 倍）"],
        ["露光", "CSV", "リング 0–3：1 s、4–7：10 s", "CSV 第 6 列"],
        ["ダーク値", "—", "100 counts", f"最小値からの推定 {dke['p1']['offset']:.0f}–{dke['p5']['offset']:.0f}（+ 背景約 {dke['p1']['bg_per_s']:.0f} counts/s）"]]
table(s, rows, 0.5, 1.2, 12.3, col_w=[1.7, 1.5, 2.8, 6.3], size=12, row_h=0.46)
textbox(s, 0.5, 5.55, 12.3, 1.4, [
    "走査：FZP の移動量 30–100 µm の 8 リング（15–50 点、計 260 点）。|k|/kc のリング中央値（0–7）= " + "、".join(f"{v:.2f}" for v in kr),
    f"使った画像：明視野 = リング 0–2（60 枚）、暗視野 = リング 6–7（95 枚）。リング 3（NA の縁）と 4–5（視野の角に直接光）は除外。外した画像 = keep[::9]（7 枚 / 18 枚）"], size=12.5, space=4)

# 4 ---- geometry & registration
s = new(); title(s, "照明の幾何と位置合わせ")
picture(s, W + "d32a/fig_geometry_32a.png", 0.35, 1.1, w=7.2, h=5.95)
textbox(s, 7.75, 1.1, 5.25, 5.9, [("h", "照明波数（試料なし像のケラレ）"),
    f"直接光が瞳を通る円盤：x_c = A d + c₀、R = {geo['R_field_um']:.1f} µm → k/kc = −x_c/R（p によらない）",
    f"FOV の曲率 κ = kc/R = {c_eff:.2f} × 1/(λ·0.75 m)。WOTF 較正はこの配置では収束せず",
    ("h", "位置合わせ"),
    "FZP X は整数パルス（2.5 µm/パルス）。検出器の値は 1.25 µm 刻みのレンズ位置から計算 → 約半数の像が 39.1 px（1.25 µm）ずれ（図 c）→ 補正",
    "残りのずれは位置の 2×2 行列（posaffine）で機械誤差として補正（次のスライド）",
    ("h", "瞳の初期値"),
    (f"コントラストのピント（瞳 − 物体の再合焦）：z = {PAR1['pupil_defocus_um']/1e3:+.2f} mm → フォーカスループで物体のずれを瞳へ：z = {PAR['pupil_defocus_um']/1e3:+.2f} mm、非点 ({PAR['pupil_astig_um'][0]/1e3:+.2f}, {PAR['pupil_astig_um'][1]/1e3:+.2f}) mm → 3 手法の初期値" if AFF2 else
     f"コントラストのピント（瞳 − 物体の再合焦）：z = {PAR['pupil_defocus_um']/1e3:+.2f} mm、非点 ({PAR['pupil_astig_um'][0]/1e3:+.2f}, {PAR['pupil_astig_um'][1]/1e3:+.2f}) mm → 3 手法の初期値" if PAR else
     f"BLIS-FPM の瞳位相の 2 次式当てはめ：z = {calb['defocus_um']/1000:.1f} mm、非点 ({calb['astig_um'][0]/1000:.2f}, {calb['astig_um'][1]/1000:.2f}) mm → 3 手法の初期値")],
    size=12, space=4)

# 4b ---- posaffine
PAFIG = W + ("fpt_pipeline/fpt_output_32a_bf_aff2/figures/fig0_position_affine.png" if AFF2 else "fpt_pipeline/fpt_output_32a_bf_aff/figures/fig0_position_affine.png")
if PAR and os.path.exists(PAFIG):
    s = new(); title(s, "位置の 2×2 行列：瞳の視差と機械的なずれの分離（posaffine）")
    picture(s, PAFIG, 0.3, 1.1, w=6.9, h=5.9)
    ms_, pp_, mm_ = PAR["measured"], PAR["pupil_predicted"], PAR["mechanical"]
    rows = [["比例ずれ", "測定", "瞳が予測", "機械的（補正）"]]
    for key_, lab_ in (("scale_col_pct", "列←X（%）"), ("scale_row_pct", "行←Y（%）"), ("col_from_row_axis_pct", "列←Y（%）"), ("row_from_col_axis_pct", "行←X（%）"), ("rotation_deg", "回転（°）"), ("shear_pct", "せん断（%）")):
        rows.append([lab_, f"{ms_[key_]:+.3f}", f"{pp_[key_]:+.3f}", f"{mm_[key_]:+.3f}"])
    table(s, rows, 7.4, 1.15, 5.6, col_w=[1.4, 1.3, 1.4, 1.5], size=11, row_h=0.33)
    ck = PAR["check"]
    textbox(s, 7.4, 3.65, 5.6, 3.4, [
        f"ずれの測定：隣り合う像の相互相関 {PAR['network']['n_pairs']} 組（閉合 {PAR['network']['closure_rms_px']:.2f} px）",
        f"高次 rms {PAR['higher_order_rms_px'][0]:.1f} / {PAR['higher_order_rms_px'][1]:.1f} px、ランダム rms {PAR['random_rms_px'][0]:.1f} / {PAR['random_rms_px'][1]:.1f} px（行 / 列）→ 機械誤差として補正",
        ("h", "毎回の確認（位置を固定して解き直す）"),
        f"(瞳 − 物体の再合焦) が予測する比例ずれ − 残したずれ：{ck['total_focus_drift_minus_retained_pct']:.3f} %",
        f"モデル像との残りの比例ずれ：{ck['residual_linear_drift_pct']:.3f} %（{ck['residual_linear_drift_max_px']:.2f} px）、残りのランダム rms {ck['random_rms_px'][0]:.2f} / {ck['random_rms_px'][1]:.2f} px",
        (f"判定：{'整合' if PAR['consistent'] else '基準超え'}（基準 0.1 %）。フォーカスループ後は、自由な瞳の多項式のデフォーカスが固定した瞳より小さく出るため（次のスライド）" if AFF2 else
         f"判定：{'整合' if PAR['consistent'] else '不整合'}（基準 0.1 % ≈ レンズ移動 100 µm で 3 px）")], size=11.5, space=3)
# 4b2 ---- focus loop
FLFIG = W + "d32a/fig_focusloop_32a.png"
if AFF2 and os.path.exists(FLFIG):
    L0 = PAR["focus_loop"]; mm1, mm2 = PAR1["mechanical"], PAR["mechanical"]; ck1, ck2 = PAR1["check"], PAR["check"]
    AQ = g("J")("d32a/af_test/pupil_annulus_quadratic.json"); zq = [f_["z_um"] / 1e3 for f_ in AQ["aff2"]["annulus_quadratic"]]
    s = new(); title(s, "物体側のピントずれを瞳へ移す（フォーカスループ）")
    picture(s, FLFIG, 0.3, 1.05, w=7.6, h=5.95)
    textbox(s, 8.1, 1.1, 5.0, 5.95, [("h", "手順（posaffine、focus_iter）"),
        "瞳をコントラストのピントに固定（デフォーカス＋非点）→ 物体だけを解く → 物体のピントずれを瞳に移し、視差と補正をやり直す",
        "  ".join(f"pass {i_}：瞳 {l_['pupil_defocus_um']/1e3:+.2f} → 物体 {l_['object_refocus']['z_um']/1e3:+.2f} mm（misfit {l_['misfit']:.4f}）" for i_, l_ in enumerate(L0)),
        ("h", "縮退"),
        "物体のデフォーカス z ＝ 瞳のデフォーカス z ＋ 視差 λzk。k に比例する視差は機械的な倍率と区別できない → misfit は変わらない",
        f"「試料にピントが合う」条件で決定。機械的な等方成分 {mm2['isotropic_pct'] - mm1['isotropic_pct']:+.2f} %（列←X {mm1['scale_col_pct']:+.2f} → {mm2['scale_col_pct']:+.2f} %）。回転・せん断は不変",
        ("h", "自由な瞳での確認"),
        f"物体の再合焦 {ck1['object_refocus']['z_um']/1e3:+.2f} → {ck2['object_refocus']['z_um']/1e3:+.2f} mm。自由な瞳のデフォーカスは 4 次多項式 {ck2['pupil']['defocus_um']/1e3:+.2f} mm、2 次だけ {min(zq):+.2f}〜{max(zq):+.2f} mm、Zernike {PBT['aff2']['zernike']['z_balanced_um']/1e3:+.2f} mm（中心に約 {AQ['aff2']['central_step_rad']:.1f} rad の段差）"],
        size=10.5, space=3)

# 4c ---- astigmatism origin
ASFIG = W + "d32a/fig_astig_origin_32a.png"
if AST and os.path.exists(ASFIG):
    ref_, kp_, nl_ = AST["reference: BLIS used in comparison (calibrated init)"], AST["drift kept (current data)"], AST["linear drift removed"]
    s = new(); title(s, "楕円状の癖の原因：瞳の非点は比例ずれの肩代わり")
    picture(s, ASFIG, 0.3, 1.1, w=8.2, h=5.9)
    AFk = g("J")("d32a/astig_test/autofocus_keep_nolin.json") if os.path.exists(W + "d32a/astig_test/autofocus_keep_nolin.json") else None
    tx = [f"以前の版の瞳：主軸方向の焦点ずれ {ref_['principal_mm'][0]:.1f} / {ref_['principal_mm'][1]:.1f} mm（{ref_['axis_deg']:.0f}°）",
          f"比例ずれを残したデータ・瞳 0 から：同じ非点（{kp_['principal_mm'][0]:.1f} / {kp_['principal_mm'][1]:.1f} mm）",
          f"比例ずれを除いたデータ・瞳 0 から：{nl_['principal_mm'][0]:.1f} / {nl_['principal_mm'][1]:.1f} mm（非点ほぼなし）。外した画像の misfit は {AST['linear drift removed misfit']['heldout']:.3f} 対 {AST['drift kept misfit']['heldout']:.3f} で区別できない"]
    if AFk:
        tx.append(f"瞳と物体の 2 次位相は入れ替え可能。和（コントラストのピント）は、ずれを残した解 z {AFk['keep0']['total_defocus_um']/1e3:+.2f} mm・非点 ({AFk['keep0']['total_astig_um'][0]/1e3:+.2f}, {AFk['keep0']['total_astig_um'][1]/1e3:+.2f})、除いた解 z {AFk['nolin0']['total_defocus_um']/1e3:+.2f} mm・非点 ({AFk['nolin0']['total_astig_um'][0]/1e3:+.2f}, {AFk['nolin0']['total_astig_um'][1]/1e3:+.2f}) mm → 光学系の非点は小さい")
    tx.append("瞳の縁のぼけは縦横ほぼ同じ → コヒーレンスの異方性は主因ではない")
    textbox(s, 8.7, 1.15, 4.4, 5.9, tx, size=11.5, space=5)

# 5 ---- dr and FOV strength
s = new(); title(s, "Δr（瞳半径）と FOV の強さ c")
picture(s, W + "d32a/fig_fov_dr_32a.png", 0.35, 1.15, w=7.4, h=5.0)
rows = [["BLIS-FPM（明視野、7 枚除外）", "当てはめ", "外した 7 枚"]]
for k, lab in [("reg2_dr250_c0", "Δr 250 nm・平面波"), ("reg2_dr250_c1", "Δr 250 nm・FOV"), ("reg2_dr200_c0", "Δr 200 nm・平面波"), ("reg2_dr200_c1", "Δr 200 nm・FOV")]:
    rows.append([lab, f3(dr[k]["fitted"]), f3(dr[k]["heldout"])])
table(s, rows, 7.95, 1.25, 5.0, col_w=[2.6, 1.1, 1.3], size=11.5, row_h=0.38)
rows = [["c"] + list(fs.keys()), ["外した 7 枚"] + [f3(v["heldout"]) for v in fs.values()]]
table(s, rows, 7.95, 3.4, 5.0, col_w=[1.6] + [1] * len(fs), size=11, row_h=0.38, first_col_bold=True)
textbox(s, 7.95, 4.35, 5.0, 2.6, ["Δr 250 nm（kc 2.0）の方がよく合う → 以後 Δr = 250 nm",
    f"c = 0.5–1.25 はほぼ横ばい、ケラレからの c = {c_eff:.2f} もこの範囲 → 公称 c = 1（p = 0.75 m）を FOV モデルに使用",
    "平面波（c = 0）と c ≥ 1.5 は明らかに悪い"], size=12.5, space=5)

# 6.. ---- comparisons
def comp_slide(key, ttl, dark=False, fig="fig_images_FOV_c1.png"):
    if key not in C: return
    s = new(); title(s, ttl)
    picture(s, FIG(key, fig), 0.3, 1.08, w=6.55, h=6.0)
    Sd = C[key]
    if not dark:
        rows = [["手法", "反復\n平面波 / FOV", "外した画像\n平面波", "外した画像\nFOV", "スポーク\n平面波", "スポーク\nFOV"]]
        for m, mn in M:
            rows.append([mn, f"{it(key, m, PL)} / {it(key, m, FV)}", f3(ho(key, m, PL)), f3(ho(key, m, FV)), sp(key, m, PL), sp(key, m, FV)])
        cw = [2.0, 1.35, 1.2, 1.2, 1.05, 1.05]
    else:
        rows = [["手法", "モデル", "外した 明視野", "外した 暗視野", "スポーク"]]
        for _, mlab, mj in MOD:
            for m, mn in M:
                rows.append([mn, mj, f3(ho(key, m, mlab, "bright")), f3(ho(key, m, mlab, "dark")), sp(key, m, mlab)])
        cw = [2.0, 1.4, 1.2, 1.2, 1.0]
    rh = 0.42 if not dark else 0.31
    table(s, rows, 7.0, 1.2, 6.0, col_w=cw, size=10.5, row_h=rh)
    y = 1.2 + rh * len(rows) + 0.25
    txt = [f"学習 {Sd['images_train']} 枚・外した画像 {Sd['images_validation']} 枚" + (f"（暗視野 {Sd['dark_images_train']} / {Sd['dark_images_validation']} 枚）" if dark else ""),
           "misfit：0.3–3.5 µm⁻¹ の強度スペクトルの相対誤差（帯域制限しない物体で計算。1 = 予測なし）",
           "スポーク：36 本の変調が SNR > 3 を保つ最高周波数（µm⁻¹）"]
    if dark:
        txt.append("BLIS-FPM は明視野の解から開始（平面波は初期値 0 だと暗視野の予測が 0/0）。EPRY は外した 18 枚（暗視野 11 枚）の平均で早いスイープが選ばれる")
    textbox(s, 7.0, y, 6.0, 7.0 - y, txt, size=11.5, space=4)

comp_slide(MAIN["bf512"], "明視野（リング 0–2）・中央 512 px：FOV モデル")
comp_slide(MAIN["bf512"], "明視野・中央 512 px：平面波モデル", fig="fig_images_planewave.png")
if MAIN["bf512"] in C:
    s = new(); title(s, "明視野・中央 512 px：星の中心の拡大とスポーク変調")
    picture(s, FIG(MAIN["bf512"], "fig_zoom.png"), 0.4, 1.1, w=7.4)
    picture(s, FIG(MAIN["bf512"], "fig_spokes.png"), 0.4, 5.3, h=1.75)
    textbox(s, 8.6, 1.15, 4.4, 5.8, ["拡大は星の中心 5.1 µm 角の位相（帯域フィルターなし、モデルごとに共通のグレースケール）",
        f"スポークの限界：{min(spk(MAIN['bf512'])):.2f}–{max(spk(MAIN['bf512'])):.2f} µm⁻¹ で手法の差は小さい",
        "DIP は滑らか、EPRY・BLIS-FPM は細かい粒状の模様 → 分解能の差ではなく、雑音・誤差の現れ方の違い",
        "星の中心は内側（r < 85 px）と外側で約 5 px ずれている → 中心を 2 領域で別々に取って解析（1 つにすると約 2.1 µm⁻¹ で見かけの頭打ち）"], size=12.5, space=6)
comp_slide(MAIN["bf1000"], "明視野・全視野 1000 px：FOV モデル")
if MAIN["bf1000"] in C:
    s = new(); title(s, "明視野・全視野 1000 px：平面波モデルと星の中心")
    picture(s, FIG(MAIN["bf1000"], "fig_images_planewave.png"), 0.3, 1.08, w=6.55, h=6.0)
    picture(s, FIG(MAIN["bf1000"], "fig_zoom.png"), 7.0, 1.15, w=6.0)
    picture(s, FIG(MAIN["bf1000"], "fig_spokes.png"), 7.1, 4.55, w=5.8)
comp_slide(MAIN["df512"], "明視野 + 暗視野（リング 0–2 + 6–7）・中央 512 px：FOV モデル", dark=True)
comp_slide(MAIN["df1000"], "明視野 + 暗視野・全視野 1000 px：FOV モデル", dark=True)


# ---- before / after
def ba_rows(g_, dark_):
    kn_, ko_ = MAIN[g_], OLD[g_]
    if kn_ not in C or ko_ not in C: return None
    rows = [["手法（FOV）", "外した画像 前→後"] + (["暗視野 前→後"] if dark_ else []) + ["スポーク 前→後", "方向の偏り 前→後"]]
    for m, mn in M:
        xo, xn = rr(ko_, m, FV), rr(kn_, m, FV)
        if xo is None or xn is None: continue
        lb_ = f"{ML[m]}, {FV}"
        ao = ANI.get(ko_, {}).get(lb_, {}).get("score"); an = ANI.get(kn_, {}).get(lb_, {}).get("score")
        row = [mn, f"{f3(xo['misfit_unfiltered']['heldout']['all'])} → {f3(xn['misfit_unfiltered']['heldout']['all'])}"]
        if dark_: row.append(f"{f2(xo['misfit_unfiltered']['heldout'].get('dark'))} → {f2(xn['misfit_unfiltered']['heldout'].get('dark'))}")
        row += [f"{f2(xo.get('spoke_snr3_um_inv'))} → {f2(xn.get('spoke_snr3_um_inv'))}", f"{f2(ao)} → {f2(an)}"]
        rows.append(row)
    return rows
if AFF:
    s = new(); title(s, "補正前後の比較（FOV モデル）", sub="前 = 以前の版（瞳に非点）、後 = " + ("この版（2×2 行列の補正＋フォーカスループ）" if AFF2 else "機械的な像ずれを補正した版"))
    y = 1.3
    for g_, lab_g, dark_ in (("bf512", "明視野・512 px", False), ("bf1000", "明視野・1000 px", False), ("df512", "暗視野を含む・512 px", True), ("df1000", "暗視野を含む・1000 px", True)):
        rows = ba_rows(g_, dark_)
        if not rows: continue
        x0 = 0.4 if g_ in ("bf512", "df512") else 6.85
        if g_ == "df512": y = 4.35
        textbox(s, x0, y - 0.05, 6.0, 0.35, [lab_g], size=12, bullets=False)
        for p in s.shapes[-1].text_frame.paragraphs:
            for q in p.runs: q.font.bold = True
        cw = [1.55, 1.35, 1.0, 1.05, 1.05] if dark_ else [1.6, 1.6, 1.4, 1.4]
        table(s, rows, x0, y + 0.3, 6.0, col_w=cw, size=9.5, row_h=0.3)
    textbox(s, 0.4, 6.75, 12.5, 0.4, ["方向の偏り：星の位相スペクトルの −22° / 68° 方向の強さの比の対数 rms（0.5–3.5 µm⁻¹、0 = 等方。星自体の方向差も含む）"], size=10, bullets=False, color=GREY)
    # per-method drift checks
    rows = [["手法", "モデル", "瞳 z / 非点\n(mm)", "物体の\n再合焦 z", "コントラストの\nピント z / 非点", "(A) 瞳だけ\n− 残した (%)", "(B) ピント\n− 残した (%)", "(C) DIP\n学習ずれ (%)", "(D) モデル\nとデータ (%)"]]
    for mod, mlab, mj in MOD:
        for m, mn in M:
            x = rr(MAIN["bf512"], m, mlab)
            if x is None or not x.get("pupil_drift_check"): continue
            c_ = x["pupil_drift_check"]; tr_ = c_.get("learned_shift_linear_trend_pct")
            rows.append([mn, mj, f"{c_['pupil_defocus_um']/1e3:+.2f} / ({c_['pupil_astig_um'][0]/1e3:+.2f}, {c_['pupil_astig_um'][1]/1e3:+.2f})", f"{c_['object_refocus']['z_um']/1e3:+.2f}",
                         f"{c_['total_defocus_um']/1e3:+.2f} / ({c_['total_astig_um'][0]/1e3:+.2f}, {c_['total_astig_um'][1]/1e3:+.2f})", f"{c_['pupil_drift_minus_kept_pct']:.3f}",
                         f"{c_['total_drift_minus_kept_pct']:.3f}", "—" if tr_ is None else f"{tr_:.3f}",
                         "—" if c_.get("model_residual_linear_drift_pct") is None else f"{c_['model_residual_linear_drift_pct']:.3f}"])
    CK = [x["pupil_drift_check"] for x in C[MAIN["bf512"]]["results"].values() if x.get("pupil_drift_check")]
    if len(rows) > 1:
        s = new(); title(s, "各再構成の瞳が予測する比例ずれの確認（明視野・512 px）")
        table(s, rows, 0.35, 1.15, 12.6, col_w=[1.55, 1.05, 1.75, 1.0, 1.85, 1.35, 1.35, 1.35, 1.35], size=9.5, row_h=0.4)
        textbox(s, 0.4, 1.25 + 0.4 * len(rows) + 0.15, 12.5, 2.2, [
            "(A) 瞳だけが予測する比例ずれ − データに残した比例ずれ（像を動かすのは瞳だけ）。(B) コントラストのピント（瞳 − 物体の再合焦）の比例ずれ − 残した比例ずれ",
            "(C) DIP が学習した画像ごとのずれの比例成分。(D) モデル像とデータ像の相互相関から求めた残りの比例ずれ（位置が合っているかの直接の確認）。基準はどれも 0.1 %",
            (lambda zt, ast, dB: f"非点はどの再構成でも {max(ast):.2f} mm 以下。デフォーカスは手法で {min(zt):+.2f}〜{max(zt):+.2f} mm → 機械的な倍率（等方部分）に最大 {max(dB):.2f} % の不確かさ。回転・せん断・X/Y の差は影響を受けない")(
                [c_["total_defocus_um"] / 1e3 for c_ in CK], [max(map(abs, c_["total_astig_um"])) / 1e3 for c_ in CK], [c_["total_drift_minus_kept_pct"] for c_ in CK])], size=11, space=3)

# ---- focus loop: before / after, resolution
if AFF2:
    s = new(); title(s, "フォーカスループの前後（FOV モデル）", sub="前 = 2026-09-27 版（瞳 z " + f"{PAR1['pupil_defocus_um']/1e3:+.2f} mm）、後 = この版（瞳 z {PAR['pupil_defocus_um']/1e3:+.2f} mm）")
    y = 1.3
    for g_, lab_g, dark_ in (("bf512", "明視野・512 px", False), ("bf1000", "明視野・1000 px", False), ("df512", "暗視野を含む・512 px", True), ("df1000", "暗視野を含む・1000 px", True)):
        rows = [["手法（FOV）", "外した画像 前→後", "スポーク 前→後", "物体の再合焦 (mm)"]]
        for m, mn in M:
            xo, xn = rr(PREV[g_], m, FV), rr(MAIN[g_], m, FV)
            rows.append([mn, f"{f3(xo['misfit_unfiltered']['heldout']['all'])} → {f3(xn['misfit_unfiltered']['heldout']['all'])}", f"{f2(xo['spoke_snr3_um_inv'])} → {f2(xn['spoke_snr3_um_inv'])}",
                         f"{xo['pupil_drift_check']['object_refocus']['z_um']/1e3:+.2f} → {xn['pupil_drift_check']['object_refocus']['z_um']/1e3:+.2f}"])
        x0 = 0.4 if g_ in ("bf512", "df512") else 6.85
        if g_ == "df512": y = 4.2
        textbox(s, x0, y - 0.05, 6.0, 0.35, [lab_g], size=12, bullets=False)
        for p_ in s.shapes[-1].text_frame.paragraphs:
            for q_ in p_.runs: q_.font.bold = True
        table(s, rows, x0, y + 0.3, 6.0, col_w=[1.6, 1.55, 1.35, 1.5], size=9.5, row_h=0.3)
    textbox(s, 0.4, 6.6, 12.5, 0.5, ["外した画像の misfit はほぼ不変（縮退）。EPRY・BLIS-FPM の物体の再合焦量が約 +1.1 mm → ほぼ 0 になり、スポーク分解能が上がった"], size=11, bullets=False)
    for f_, ttl_, tx_ in (("fig_focusloop_zoom_32a_512.png", "再構成像：フォーカスループの前後（星の中心 5.1 µm 角、中央 512 px）",
                           ["上 2 段：明視野、下 2 段：暗視野を含む。各 2 段の上がループ前（破線の枠）、下がループ後（実線の枠）",
                            "FOV モデル、帯域フィルターなし、再構成したまま。同じデータの中では前後・4 手法で共通のグレースケール",
                            "数字：スポーク分解能（µm⁻¹）、物体の再合焦量（0 = ピントが合っている）、その手法が求めた瞳のデフォーカス",
                            "段の見出しの瞳 z（−1.52 / −2.72 mm）は全手法共通の初期値。EPRY はそのまま、DIP・BLIS-FPM は瞳を更新",
                            "明視野の EPRY・BLIS-FPM：ループ後は中心近くの細いスポークがはっきり見える（再合焦量 約 +1.1 → ほぼ 0 mm）",
                            "DIP はループの前後で大きくは変わらない。暗視野を含む DIP（最適反復）・BLIS-FPM には斑点状の模様が残る"]),
                          ("fig_focusloop_field_32a_1000.png", "再構成像：フォーカスループの前後（全視野 1000 px）",
                           ["並びは前のスライドと同じ（上 2 段：明視野、下 2 段：暗視野を含む）", "中央 512 px 全体と、1000 px の星の中心の拡大は、成果物フォルダの figures/ にあります（fig_focusloop_{zoom,field}_32a_{512,1000}.png）"])):
        if os.path.exists(W + "d32a/" + f_):
            s = new(); title(s, ttl_)
            picture(s, W + "d32a/" + f_, 0.3, 1.02, h=6.0)
            textbox(s, 7.25, 1.2, 5.85, 5.8, tx_, size=12, space=6)
    MF, FF = W + "d32a/frc/fig_mtf_signed_aff_vs_aff2_512.png", W + "d32a/frc/fig_frc_aff_vs_aff2_512.png"
    if RES2 and os.path.exists(MF) and os.path.exists(FF):
        s = new(); title(s, "分解能：フォーカスループの前後（中央 512 px、FOV）")
        picture(s, MF, 0.3, 1.05, w=6.4); picture(s, FF, 6.75, 1.05, w=6.3)
        mt = lambda s_, m: RES2[f"MTF10|512|{s_}|{ML[m]}, FOV model"]
        fr = lambda s_, m: RES2[f"FRC_halfbit|as reconstructed|{s_}|{ML[m]}"]
        fq = lambda d_: ("≥" if d_["no_crossing_below_support_limit"] else "") + f"{d_['q']:.2f}"
        rows = [["手法", "MTF 10 % 明視野\n前 → 後（前・再合焦）", "MTF 10 % 暗視野を含む\n前 → 後", "FRC 明視野\n前 → 後", "FRC 暗視野を含む\n前 → 後"]]
        for m, mn in M:
            rows.append([mn, f"{f2(mt('BF', m)['aff'])} → {f2(mt('BF', m)['aff2'])}（{f2(mt('BF', m)['aff_refocused'])}）", f"{f2(mt('BF+DF', m)['aff'])} → {f2(mt('BF+DF', m)['aff2'])}",
                         f"{fq(fr('BF', m)['aff'])} → {fq(fr('BF', m)['aff2'])}", f"{fq(fr('BF+DF', m)['aff'])} → {fq(fr('BF+DF', m)['aff2'])}"])
        table(s, rows, 0.4, 4.75, 8.6, col_w=[1.6, 2.2, 1.7, 1.5, 1.6], size=9.5, row_h=0.33)
        textbox(s, 9.2, 4.75, 3.9, 2.3, ["ループ後の EPRY・BLIS-FPM は、前の版の物体を後から再合焦した MTF・FRC に、後処理なしで一致",
            f"暗視野の帯域（3.9–4.6 µm⁻¹）：反転はなくなったが、正の値は雑音程度（EPRY {mt('BF+DF', 'EPRY')['df_band_aff2']:+.2f}、BLIS {mt('BF+DF', 'BLIS')['df_band_aff2']:+.2f}）→ 暗視野による分解能の向上は確認できない（単位 µm⁻¹）"], size=10.5, space=4)

# ---- limitations
s = new(); title(s, "制限事項と今後の課題")
textbox(s, 0.6, 1.2, 12.1, 5.8, [
    "**レンズと検出器の位置の食い違い = 像のずれ**（平行ビーム・対物走査の特徴）。今後は検出器の値を FZP に送る整数パルスから計算（検出器 X = −527.75 × FZP X）することを推奨",
    f"**機械的な比例ずれ**（列←X 約 {PAR['mechanical']['scale_col_pct']:+.1f} %、FZP Y 軸と検出器軸の角度 約 {abs(np.degrees(PAR['mechanical']['col_from_row_axis_pct'] / 100)):.1f}°）：瞳の非点と取り違えやすい → posaffine で毎回補正・確認。検出器 X の追従倍率とステージ軸の向きの較正を推奨",
    "**FZP の 0 次光**：リング 0–1 の比の像に円弧状の干渉縞（輪帯板と同じ形）。OSA で 0 次光を遮るか、順モデルに 0 次光の項を入れると改善が見込める",
    "**暗視野**：10 s 露光でも画素あたりの SNR < 1（雑音の分散 = コントラストの 2 乗の " + f"{nz6:.1f} / {nz7:.1f} 倍）。暗視野像の位置は明視野の 2×2 行列の外挿（暗視野どうしの相互相関は不安定）。露光の延長・積算が必要",
    "**照明**：スリットで制限された帯状（強い部分の幅 約 10 µm）。全視野では上下の照明が弱く、misfit を悪くする",
    ("**ピント**：物体側のずれを瞳へ移し（フォーカスループ）、どの手法の物体もピントが合った。割り振りはデータだけでは決まらない（視差と倍率の縮退）→ 試料を光軸方向に既知量動かす測定で瞳のデフォーカスと倍率を独立に決められる" if AFF2 else
"**ピント**：コントラストのピントは手法により 1–3 mm の幅 → 機械的な倍率（等方部分）に最大約 0.2 % の不確かさ。試料を光軸方向に既知量動かす測定で決められる"),
    "**p = 0.75 m は未実測**：照明波数はケラレから較正したが、FOV の曲率 c は 0.5–1.25 で区別できない",
    "**Δr**：データは Δr 250 nm 相当。使った FZP の仕様の確認をお願いします"], size=14, space=8)

# ---- files
s = new(); title(s, "出力ファイル", sub="成果物のまとめフォルダ：FPT32a_成果物まとめ_20260927/")
textbox(s, 0.6, 1.5, 12.1, 5.5, [
    "`32a_report.md`：レポート（日本語）。数値はすべて JSON から自動生成",
    "`figures/`：本スライドの図（すべて帯域フィルターなし）",
    "`tiff/<条件>/<手法>_<モデル>_{transmission, phase_rad, pupil_phase_rad}.tif`：再構成（物理フレーム、帯域フィルターなし）",
    "`pipeline_BLIS-FPM/`：fpt_pipeline（全 60 枚）の出力（results の TIFF はパイプラインの仕様で帯域制限あり）",
    "`json/`：比較・スポーク・較正・位置合わせの JSON",
    "`scripts/`：32a 用の補助スクリプト、実行シェル、各パイプラインの設定ファイル",
    "`pipeline_changes/`：fpt_pipeline への追加（fptrecon/posaffine.py、run_pipeline.py の posaffine ステップ）。以前の版のレポート・図・TIFF は `before_posaffine/`"]
    + (["`before_focusloop/`：フォーカスループ前の版（2026-09-27 版）。`supplement_focus_in_pupil/`：フォーカスループの解析と分解能の前後比較"] if AFF2 else []), size=14, space=8)

out = OUT + "FPT32a_再構成まとめ.pptx"; prs.save(out); print("slides", n, out)
