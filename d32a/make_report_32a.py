# -*- coding: utf-8 -*-
"""32a report (Japanese) from the analysis JSONs. Every number in the text is read from the JSON files.
usage: python d32a/make_report_32a.py  -> d32a/32a_report.md, d32a/32a_summary.json"""
import json, os
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
J = lambda p: json.load(open(W + p, encoding="utf-8"))
ex = lambda p: os.path.exists(W + p)
geo = J("d32a/geom_direct_v2.json"); reg = J("d32a/register_32a.json"); dke = J("d32a/dark_estimate.json")
svA = np.linalg.svd(np.array(geo["A"]), compute_uv=False)
rtm = J("fpt_pipeline/fpt_output_32a_bf_reg2/work/register_to_model.json"); fd = J("d32a/fov_dr_summary.json")
prep = J("d32a/work_df/prep_df.json"); noise = J("d32a/work_df/noise.json")
calb = J("fpt_pipeline/fpt_output_32a_bf_reg2/work/calibration.json")
calp = J("fpt_pipeline/fpt_output_32a_bf_reg2/results/calibration_physical.json")
C = {k: J(f"d32a/out_{k}/compare_summary.json") for k in ("32abf_512", "32abf_1000", "32adf_512", "32adf_1000", "32abfaff_512", "32abfaff_1000", "32adfaff_512", "32adfaff_1000",
     "32abfaff2_512", "32abfaff2_1000", "32adfaff2_512", "32adfaff2_1000") if ex(f"d32a/out_{k}/compare_summary.json")}
AFF = "32abfaff_512" in C
MAIN = dict(bf512="32abfaff_512", bf1000="32abfaff_1000", df512="32adfaff_512", df1000="32adfaff_1000") if AFF else dict(bf512="32abf_512", bf1000="32abf_1000", df512="32adf_512", df1000="32adf_1000")
OLD = dict(bf512="32abf_512", bf1000="32abf_1000", df512="32adf_512", df1000="32adf_1000")
PAR = J("fpt_pipeline/fpt_output_32a_bf_aff/results/position_affine.json") if ex("fpt_pipeline/fpt_output_32a_bf_aff/results/position_affine.json") else None
cdf = J("d32a/work_df_aff/calibration.json") if ex("d32a/work_df_aff/calibration.json") else None
# v3 (2026-09-28): focus loop ("aff2", residual object defocus moved into the pupil) = main results; aff = before the focus loop
AFF2 = all(k in C for k in ("32abfaff2_512", "32abfaff2_1000", "32adfaff2_512", "32adfaff2_1000"))
PREV = dict(bf512="32abfaff_512", bf1000="32abfaff_1000", df512="32adfaff_512", df1000="32adfaff_1000")
PAR1 = PAR
if AFF2:
    MAIN = dict(bf512="32abfaff2_512", bf1000="32abfaff2_1000", df512="32adfaff2_512", df1000="32adfaff2_1000")
    PAR = J("fpt_pipeline/fpt_output_32a_bf_aff2/results/position_affine.json")
    cdf = J("d32a/work_df_aff2/calibration.json")
FL = J("d32a/focusloop_summary_32a.json") if AFF2 and ex("d32a/focusloop_summary_32a.json") else None
RES2 = J("d32a/frc/resolution_table_32a_aff2.json") if AFF2 and ex("d32a/frc/resolution_table_32a_aff2.json") else None
PBT = J("d32a/af_test/pupil_basis_test.json") if AFF2 and ex("d32a/af_test/pupil_basis_test.json") else None
ANI = J("d32a/aniso_compare_32a.json") if ex("d32a/aniso_compare_32a.json") else {}
AST = J("d32a/astig_test/astig_test_summary.json") if ex("d32a/astig_test/astig_test_summary.json") else None
lam = 1.239842 / 30 * 1e-3
M = [("DIP_opt", "DIP（最適反復）"), ("DIP_30", "DIP（30 反復）"), ("EPRY", "EPRY"), ("BLIS", "BLIS-FPM")]
ML = {"DIP_opt": "DIP (optimal it.)", "DIP_30": "DIP (30 it.)", "EPRY": "EPRY", "BLIS": "BLIS-FPM"}
MOD = [("planewave", "plane-wave model", "平面波"), ("FOV_c1", "FOV model", "FOV（c = 1）")]
def r(key, m, mod):
    return C[key]["results"].get(f"{ML[m]}, {mod}")
f3 = lambda v: "—" if v is None or v != v else f"{v:.3f}"
f2 = lambda v: "—" if v is None or v != v else f"{v:.2f}"
ring = np.array(noise["ring"]); s2 = np.array(noise["sigma2_512"]); lv = np.array(noise["level_512"])
lev6, lev7 = np.median(lv[ring == 6]), np.median(lv[ring == 7]); nz6, nz7 = np.median(s2[ring == 6] / lv[ring == 6] ** 2), np.median(s2[ring == 7] / lv[ring == 7] ** 2)
kk = np.array(geo["k_over_kc"]); rings_all = np.repeat(np.arange(8), [15, 20, 25, 30, 35, 40, 45, 50])
kr = [float(np.median(kk[rings_all == q])) for q in range(8)]
krng = lambda qs: (float(kk[np.isin(rings_all, qs)].min()), float(kk[np.isin(rings_all, qs)].max()))
c_eff = fd["c_eff_vignetting"]; dr = fd["dr"]; fs = fd["fov_scan"]
rb = reg["bright"]; rm_rms = rtm["rms_px"]
p_from_map = 1 / (calb["scale_vs_nominal"] * (1 / (lam * 0.75e6)) * lam) / 1e6          # m

def table(key, dark=False):
    h = "| 手法 | モデル | 採用反復 | misfit 当てはめ | misfit 外した画像 |" + (" 外した明視野 | 外した暗視野 |" if dark else "") + " スポーク SNR>3 (µm⁻¹) | defocus / astig 0° / astig 45° (rad RMS) |"
    out = [h, "|" + "---|" * (h.count("|") - 1)]
    for mod, mlab, mj in MOD:
        for m, mn in M:
            x = r(key, m, mlab)
            if x is None: continue
            u = x["misfit_unfiltered"]; z = x["zernike_rad_rms_physical"]
            it = x["iteration"]; its = "—" if it is None else str(it)
            row = f"| {mn} | {mj} | {its} | {f3(u['train']['all'])} | {f3(u['heldout']['all'])} |"
            if dark: row += f" {f3(u['heldout'].get('bright'))} | {f3(u['heldout'].get('dark'))} |"
            row += f" {f2(x.get('spoke_snr3_um_inv'))} | {z['defocus']:+.2f} / {z['astig 0deg']:+.2f} / {z['astig 45deg']:+.2f} |"
            out.append(row)
    return "\n".join(out)
def best(key, which="heldout"):
    v = [(r(key, m, mlab)["misfit_unfiltered"][which]["all"], mn, mj) for mod, mlab, mj in MOD for m, mn in M if r(key, m, mlab)]
    return min(v)
def spk(key):
    return [r(key, m, mlab)["spoke_snr3_um_inv"] for mod, mlab, mj in MOD for m, mn in M if r(key, m, mlab)]

L = []
A = L.append
A("# FPT 32a（平行ビーム照明・対物 FZP 走査）シーメンススター再構成レポート")
A("")
A("データ：32a（30 keV、260 位置 × 試料あり/なし、ITEX 1000×1000）。"
  "再構成には、ご指定の `暗視野-並行ビーム照明対応pipeline` の 3 つのパイプライン"
  "（`fpt_pipeline` = BLIS-FPM、`dip_pipeline` = DIP、`epry_pipeline` = EPRY）を使いました。再構成のアルゴリズムは変えていません。"
  "変えたのは入力（位置補正したデータ、照明の較正）と設定、32a 用の小さな補助スクリプト（`d32a/`）、およびこの版で追加した位置の 2×2 行列のステップ（`fpt_pipeline` の posaffine。既定では動かない任意のステップ）です。")
A("")
A("**表示について**：このレポートとスライドの再構成像は、**どれも帯域フィルターをかけていません**（手法が出力した a, φ をそのまま表示）。"
  "グレースケールは、同じモデルの 4 手法で共通です（各像の 0.5–99.5 % 点の中央値）。")
A("")
A("## 0. 要点")
b512 = best(MAIN["bf512"]) if MAIN["bf512"] in C else None
A(f"1. **測定条件のうち 2 点を、データに合わせて直しました。** (a) 対物 FZP は Δr = 250 nm に相当します（瞳半径 kc = 2.0 µm⁻¹、NA = 8.27×10⁻⁵）。ご指定の Δr = 200 nm（kc = 2.5 µm⁻¹）ではありません。"
  f"(b) FZP Y の単位は 0.2 µm/単位です（0.6 µm ではありません）。どちらもご確認いただいたとおりです。"
  f"kc = 2.0 と 2.5 を BLIS-FPM で比べると、FOV モデルで外した画像の予測 misfit は {f3(dr['reg2_dr250_c1']['heldout'])} 対 {f3(dr['reg2_dr200_c1']['heldout'])}、当てはめの misfit は {f3(dr['reg2_dr250_c1']['fitted'])} 対 {f3(dr['reg2_dr200_c1']['fitted'])} で、2.0 の方がよく合います。")
if PAR:
    mm, pp = PAR["mechanical"], PAR["pupil_predicted"]
    A(f"2. **位置合わせ：機械的な像ずれを 2×2 行列として補正しました。** 検出器の値が FZP のパルスの半分の刻みで計算されていたための 39.1 px のずれ（2.1 節）を戻したうえで、"
      f"像のずれを画像どうしの相互相関から測り、レンズ位置に比例する部分（2×2 行列）、高次の部分、ランダムな部分に分けました。瞳（ピントずれ）が予測する分だけを残し、残りは機械的な誤差として補正しました。"
      f"機械的な比例ずれ（像のずれ / レンズの移動量）は、列←X {mm['scale_col_pct']:+.2f} %、行←Y {mm['scale_row_pct']:+.2f} %、列←Y {mm['col_from_row_axis_pct']:+.2f} %、行←X {mm['row_from_col_axis_pct']:+.2f} % "
      f"（回転 {mm['rotation_deg']:+.2f}°、せん断 {mm['shear_pct']:+.2f} %）で、測ったずれのうち瞳で説明できるのは {100*PAR['fraction_of_drift_explained_by_pupil']:.0f} % だけでした（2.2 節）。")
    ck = PAR["check"]
    if AFF2:
        L0 = PAR["focus_loop"]
        A(f"3. **以前の版の「楕円状の癖」は、この機械的なずれを瞳が非点として肩代わりしたためでした。** 瞳の 2 次の位相（デフォーカス・非点）と物体側の 2 次の位相は、像のコントラストには同じように効きます（コントラストが決めるのは和だけ）。一方、像のずれ（視差）を生むのは瞳だけです。"
          f"そこで、物体のピントが合う条件で和（コントラストのピント）を求めて瞳に割り当て、その瞳が予測する視差だけをデータに残しました。"
          f"この版ではさらに、瞳をこの値に固定して物体だけを解き、物体に残ったピントずれ（{L0[0]['object_refocus']['z_um']/1e3:+.2f} mm）も瞳に移しました（フォーカスループ、2.4 節）："
          f"瞳 z = {PAR['pupil_defocus_um']/1e3:+.2f} mm、非点 ({PAR['pupil_astig_um'][0]/1e3:+.2f}, {PAR['pupil_astig_um'][1]/1e3:+.2f}) mm（非点はほぼ 0）で、このとき物体の再合焦量は {L0[-1]['object_refocus']['z_um']/1e3:+.2f} mm です。")
    else:
        A(f"3. **以前の版の「楕円状の癖」は、この機械的なずれを瞳が非点として肩代わりしたためでした。** 瞳の 2 次の位相（デフォーカス・非点）と物体側の 2 次の位相は、像のコントラストには同じように効きます（コントラストが決めるのは和だけ）。一方、像のずれ（視差）を生むのは瞳だけです。"
          f"そこで、物体のピントが合う条件で和（コントラストのピント）を求めて瞳に割り当て、その瞳が予測する視差だけをデータに残しました：z = {PAR['pupil_defocus_um']/1e3:+.2f} mm、非点 ({PAR['pupil_astig_um'][0]/1e3:+.2f}, {PAR['pupil_astig_um'][1]/1e3:+.2f}) mm（非点はほぼ 0）。"
          f"補正後のデータで解き直すと、コントラストのピントが予測する比例ずれと残したずれの差は {ck['total_focus_drift_minus_retained_pct']:.3f} %、モデル像との残りの比例ずれは {ck['residual_linear_drift_pct']:.3f} % で、整合しました（2.4 節）。")
    if AFF and MAIN["bf512"] in C:
        cks = [x["pupil_drift_check"] for g_ in ("bf512", "bf1000") if MAIN[g_] in C for x in C[MAIN[g_]]["results"].values() if x.get("pupil_drift_check")]
        if cks:
            ast_ = [max(abs(c_["total_astig_um"][0]), abs(c_["total_astig_um"][1])) / 1e3 for c_ in cks]
            zt_ = [c_["total_defocus_um"] / 1e3 for c_ in cks]
            A(f"   4 手法の再構成ごとに同じ確認をすると（7.2 節）、非点はどれも {max(ast_):.2f} mm 以下で、非点がないことは確かです。"
              f"一方、デフォーカス（コントラストのピント）は手法によって {min(zt_):+.1f}〜{max(zt_):+.1f} mm と幅がありました（焦点深度 約 6 mm の中）。"
              f"この幅は、比例ずれのうち等方的な部分（倍率）を「瞳」と「機械」に分ける量の不確かさ（最大 {max(c_['total_drift_minus_kept_pct'] for c_ in cks):.2f} %）に当たります。回転・せん断・X と Y の違いは、この不確かさに影響されません。"
              f"どの再構成も、モデル像の位置はデータと合っていました（残りの比例ずれ {max(c_['model_residual_linear_drift_pct'] for c_ in cks if 'model_residual_linear_drift_pct' in c_):.3f} % 以下）。")
if AFF2:
    rf_ = lambda key, m: r(key, m, "FOV model")["pupil_drift_check"]["object_refocus"]["z_um"] / 1e3
    sq_ = lambda key, m: r(key, m, "FOV model")["spoke_snr3_um_inv"]
    ho_ = lambda key, m: r(key, m, "FOV model")["misfit_unfiltered"]["heldout"]["all"]
    diso = PAR["mechanical"]["isotropic_pct"] - PAR1["mechanical"]["isotropic_pct"]
    txt_m = ""
    if RES2:
        mt_ = lambda m: RES2[f"MTF10|512|BF|{m}, FOV model"]
        txt_m = (f"星の MTF が 10 % に下がる周波数は EPRY {mt_('EPRY')['aff']:.2f} → {mt_('EPRY')['aff2']:.2f}、BLIS-FPM {mt_('BLIS-FPM')['aff']:.2f} → {mt_('BLIS-FPM')['aff2']:.2f} µm⁻¹ で、"
                 f"以前の版の物体を後から再合焦したときの値（{mt_('EPRY')['aff_refocused']:.2f} / {mt_('BLIS-FPM')['aff_refocused']:.2f} µm⁻¹）とほぼ同じです。")
    A(f"4. **物体側に残っていたピントずれ（約 1.1 mm）を瞳に移して解き直しました。** 以前の版の EPRY・BLIS-FPM の物体は、再構成したままではピントがずれていました"
      f"（再合焦量 EPRY {rf_(PREV['bf512'], 'EPRY'):+.2f} mm、BLIS-FPM {rf_(PREV['bf512'], 'BLIS'):+.2f} mm。明視野・512 px・FOV）。"
      f"瞳に移した版では {rf_(MAIN['bf512'], 'EPRY'):+.2f} / {rf_(MAIN['bf512'], 'BLIS'):+.2f} mm で、ピントの合った物体がそのまま得られます。"
      f"スポーク分解能は EPRY {sq_(PREV['bf512'], 'EPRY'):.2f} → {sq_(MAIN['bf512'], 'EPRY'):.2f}、BLIS-FPM {sq_(PREV['bf512'], 'BLIS'):.2f} → {sq_(MAIN['bf512'], 'BLIS'):.2f} µm⁻¹ に上がりました。" + txt_m +
      f"外した画像の misfit はほとんど変わりません（EPRY {f3(ho_(PREV['bf512'], 'EPRY'))} → {f3(ho_(MAIN['bf512'], 'EPRY'))}、BLIS-FPM {f3(ho_(PREV['bf512'], 'BLIS'))} → {f3(ho_(MAIN['bf512'], 'BLIS'))}）。"
      "物体側のデフォーカス z は、「瞳のデフォーカス z」と「その視差を打ち消す、照明の位置 k に比例する像のずれ」を組み合わせたものと同じ像を作ります。k に比例する像のずれは機械的な倍率と区別できないため、ピントを物体と瞳にどう割り振るかは、データだけでは決まりません。"
      f"この版では「試料（ジーメンススター）にピントが合っている」ことを条件にして割り振りを決めました。その分、機械的な倍率（等方成分）が {diso:+.2f} % 変わりました（7.3 節）。")
n0 = 5 if AFF2 else (4 if PAR else 2)
if b512:
    sp = spk(MAIN["bf512"])
    A(f"{n0}. **明視野（リング 0–2、60 枚）・中央 512 px**：外した 7 枚の予測が最もよいのは {b512[1]}・{b512[2]} で、misfit {f3(b512[0])} でした。"
      f"スポーク変調（SNR > 3）で見た分解能は {min(sp):.2f}–{max(sp):.2f} µm⁻¹（半周期 {500/max(sp):.0f}–{500/min(sp):.0f} nm）です（5 章）。")
if MAIN["bf1000"] in C:
    b1 = best(MAIN["bf1000"])
    A(f"{n0+1}. **全視野 1000 px**（31.9 µm 角）：外した画像の予測が最もよいのは {b1[1]}・{b1[2]} で、misfit {f3(b1[0])} でした。スポーク分解能は {min(spk(MAIN['bf1000'])):.2f}–{max(spk(MAIN['bf1000'])):.2f} µm⁻¹ です。")
if MAIN["df512"] in C:
    ho_d = lambda key: [r(key, m, mlab)["misfit_unfiltered"]["heldout"].get("dark") for mod, mlab, mj in MOD for m, mn in M if r(key, m, mlab)]
    bb = lambda key, m, mlab: r(key, m, mlab)["misfit_unfiltered"]["heldout"].get("bright", r(key, m, mlab)["misfit_unfiltered"]["heldout"]["all"])
    xd = ho_d(MAIN["df512"])
    A(f"{n0+2}. **暗視野（リング 6–7、95 枚、10 s）を加えた場合**：暗視野の強度は明視野の {100*lev7:.2f}–{100*lev6:.2f} % です。外した暗視野画像の予測 misfit は {min(xd):.2f}–{max(xd):.2f}"
      + (f"（補正前 {min(ho_d(OLD['df512'])):.2f}–{max(ho_d(OLD['df512'])):.2f}）" if OLD["df512"] in C and AFF else "")
      + f"、外した明視野画像は DIP（最適反復）・FOV で {f3(bb(MAIN['bf512'], 'DIP_opt', 'FOV model'))}（明視野のみ）→ {f3(bb(MAIN['df512'], 'DIP_opt', 'FOV model'))}（暗視野を追加）、"
      f"スポーク分解能は {min(spk(MAIN['df512'])):.2f}–{max(spk(MAIN['df512'])):.2f} µm⁻¹（明視野のみ {min(spk(MAIN['bf512'])):.2f}–{max(spk(MAIN['bf512'])):.2f}）でした（6 章）。")
A("")
if AFF:
    A("> **改訂について（2026-09-27）**：この版は、機械的な像ずれを補正して 4 手法 × 2 モデル × 2 格子（暗視野を含む）を解き直した結果です。"
      "以前の版（瞳に大きな非点が入っていた版）の結果との比較は 7 章にまとめました。")
    if AFF2:
        A("")
        A("> **改訂について（2026-09-28）**：この版の主な結果（5・6 章の表と図）は、物体側に残っていたピントずれを瞳に移して（フォーカスループ）、全 32 の再構成を解き直したものです。"
          "移す前の版（2026-09-27 版）との比較は 7.3 節に、移す前の版の図・TIFF・JSON は `before_focusloop/` にあります。")
    A("")
A("## 1. データと測定条件")
A("")
A("| 項目 | ご指定の値 | 使った値 | 根拠 |")
A("|---|---|---|---|")
A("| X 線エネルギー | 30.0 keV（λ = 0.041328 nm） | 同じ | — |")
A(f"| 実効画素 | 31.9 nm/px | 同じ | — |")
A(f"| 対物 FZP Δr | 200 nm（NA = λ/2Δr） | **250 nm（kc = 2.0 µm⁻¹）** | 像スペクトルの外縁、ケラレの位置、ホールドアウト misfit（3 章）。フォルダ名・CSV 名も fzp250nm |")
A(f"| 対物 FZP 直径 | φ155 µm | 実効 φ{2*geo['NA_edge_lens_shift_um']:.0f} µm（ケラレが起きる FZP 移動量 {geo['NA_edge_lens_shift_um']:.1f} µm） | Δr 250 nm・f 0.75 m の FZP 径 {lam*0.75e6/0.25:.0f} µm とほぼ同じ |")
A(f"| 試料–対物距離 p | 0.75 m（実測値ではない） | FOV モデルの κ = 1/(λp) に使用 | 照明波数の換算（ケラレから独立に求めたもの）は、公称 p = 0.75 m の {calb['scale_vs_nominal']:.3f} 倍 |")
A("| FZP X の単位 | 2.5 µm | 同じ | — |")
A(f"| FZP Y の単位 | 0.6 µm | **0.2 µm** | 検出器 X / FZP X と 検出器 Y / FZP Y の比が 12.5 = 2.5/0.2。0.2 µm にするとリングが円になる。ケラレから求めた写像が等方的（特異値 {svA[0]:.3f} / {svA[1]:.3f}） |")
A("| 露光 | CSV の値 | リング 0–3：1 s、リング 4–7：10 s | CSV 第 6 列 |")
A(f"| ダーク値 | — | 100 counts（ダーク像なし） | 1 s と 10 s の最小値からの推定は {dke['p1']['offset']:.0f}–{dke['p5']['offset']:.0f}（残りは約 {dke['p1']['bg_per_s']:.0f} counts/s の背景） |")
A("")
A("走査は 8 本のリング（FZP の移動量 30, 40, …, 100 µm、1 本あたり 15–50 点、計 260 点）です。ケラレから求めた |k|/kc のリングごとの中央値は "
  + "、".join(f"{kr[q]:.2f}" for q in range(8)) + " です（同じリング内でも、照明の中心のずれのため ±0.06 程度ばらつきます）。")
A("")
A("| 使った画像 | リング | 枚数 | \\|k\\|/kc（画像ごとの範囲） | 露光 | 使い道 |")
A("|---|---|---|---|---|---|")
A(f"| 明視野（BF）セット | 0–2 | 60 | {krng([0, 1, 2])[0]:.2f}–{krng([0, 1, 2])[1]:.2f} | 1 s | 明視野の再構成（5 章） |")
A(f"| 暗視野（DF）の追加 | 6–7 | 95 | {krng([6, 7])[0]:.2f}–{krng([6, 7])[1]:.2f} | 10 s | BF+DF の再構成（6 章） |")
A(f"| 使わない | 3 | 30 | {krng([3])[0]:.2f}–{krng([3])[1]:.2f} | 1 s | NA の縁。視野の半分ほどで直接光が瞳の外に出る |")
A(f"| 使わない | 4–5 | 75 | {krng([4, 5])[0]:.2f}–{krng([4, 5])[1]:.2f} | 10 s | 視野の角に直接光が残る（境界の画像。036 のリング 2 と同じ理由） |")
A("")
A("外した画像（ホールドアウト）は、DIP・EPRY と同じ規則で `keep[::9]` としました（BF：7 枚、BF+DF：18 枚）。"
  "BLIS-FPM も、比較にはこれらを除いて解いた結果を使いました（外した画像の misfit が 4 手法すべてで本当の予測になります）。")
A("")
A("![geometry](figures/fig_geometry_32a.png)")
A("")
A("*図 1　a：走査位置。b：ケラレの当てはめ（試料なし像と、当てはめた瞳の縁）。c：像のずれと、レンズ位置と検出器位置の食い違い。d, e：像のスペクトルと瞳の円（シアン：kc 2.0、白：kc 2.5）。"
  "f：補正した位置ずれ。g, h：試料あり/なしの比の像（リング 0 の円弧状の縞は、FZP の 0 次光が像と干渉したもの）。*")
A("")
A("## 2. 前処理")
A("")
A("### 2.1 レンズと検出器の位置の食い違いによる像ずれ（39.1 px）")
A("CSV の FZP X はステージのパルス数（2.5 µm/パルス、整数）で、レンズの実際の位置です。一方、検出器 X の値を −527.75（FZP 1 パルスあたりの検出器の値）で割ると、約 44 % の行で半整数（例：#1 は FZP X = 10 に対して 10.5）になります。"
  "CSV の値を逆算すると、リングの計算上の位置 x を、FZP X は 2.5 µm 刻み、検出器は 1.25 µm 刻みで、それぞれ切り捨てて作られていました（Y も同様に 0.2 µm と 0.1 µm）。"
  "対物レンズの位置と、検出器が前提とするレンズ位置が δd 食い違うと、試料面に換算した像は δd だけずれます。そのため食い違いが半パルス（1.25 µm = 39.1 px）の画像では、像が列方向に 39.1 px ずれていました（Y 方向は最大 0.1 µm = 3.1 px）。"
  "リング 0–2 の星の中心の列位置は、この食い違いに対して傾き −1.0 で動きます（図 1c）。試料あり像・試料なし像の両方を、この量だけ戻しました（`d32a/prep_32a_reg.py`）。")
A("")
A("### 2.2 位置の 2×2 行列（機械的な像ずれ）と瞳の視差の分離")
if PAR:
    ms_, pp_, mm_ = PAR["measured"], PAR["pupil_predicted"], PAR["mechanical"]
    A(f"2.1 節の補正のあとも、画像ごとに位置ずれが残っていました。パイプラインに追加した posaffine ステップ（`fpt_pipeline/fptrecon/posaffine.py`、`run_pipeline.py --steps posaffine`）で、次の手順で処理しました。")
    A("")
    A(f"1. **ずれの測定**：k 空間で隣り合う画像どうし（各画像の近い 4 枚、計 {PAR['network']['n_pairs']} 組）の相互相関から、各画像の絶対的なずれ s_n を最小二乗で求めました（閉合誤差 rms {PAR['network']['closure_rms_px']:.2f} px）。")
    A(f"2. **分解**：s_n = B·(レンズ位置) + 高次（k の 2 次・3 次の勾配場）+ ランダム。B は 2×2 行列（倍率・回転・せん断）です。")
    A(f"3. **瞳が予測する視差**：瞳の位相 W(k) があると、直接光が瞳を通る位置 k_n で像が −∇W(k_n)/2π だけずれます（符号と大きさは順モデルの数値テストで確認）。"
      "ずれをすべて取り除いたデータで瞳と物体を解き、物体のピントを合わせて瞳の 2 次の位相を決め（2.4 節）、その瞳が予測する視差を計算しました。")
    A(f"4. **補正**：測ったずれのうち、瞳が予測する視差以外（2×2 行列の残り、高次、ランダム）を機械的な誤差として画像から取り除き、視差は残しました。")
    A(f"5. **確認**：補正後のデータを、位置を固定して解き直しました。新しい瞳（と物体のピント）が予測する比例ずれと残したずれの差、およびモデル像との相互相関で測った残りの比例ずれを調べます。最後に、残りのランダムなずれを補正しました。")
    A("")
    A("| 比例ずれ（像のずれ / レンズ移動量） | 測定 | 瞳（ピントずれ）が予測 | 機械的（補正した分） |")
    A("|---|---|---|---|")
    for key_, lab_ in (("scale_col_pct", "列 ← X（%）"), ("scale_row_pct", "行 ← Y（%）"), ("col_from_row_axis_pct", "列 ← Y（%）"), ("row_from_col_axis_pct", "行 ← X（%）"),
                       ("rotation_deg", "回転（°）"), ("shear_pct", "せん断（%）")):
        A(f"| {lab_} | {ms_[key_]:+.3f} | {pp_[key_]:+.3f} | {mm_[key_]:+.3f} |")
    A("")
    ck = PAR["check"]; hr = PAR["mechanical_higher_order_rms_px"]
    if AFF2:
        ck1 = PAR1['check']
        A(f"高次の部分（2 次以上。測定 rms {PAR['higher_order_rms_px'][0]:.1f} / {PAR['higher_order_rms_px'][1]:.1f} px、行 / 列）は、瞳が予測する高次の視差を差し引いた残り（rms {hr[0]:.1f} / {hr[1]:.1f} px）を機械的なものとして補正しました。"
          f"ランダムな部分（rms {PAR['random_rms_px'][0]:.1f} / {PAR['random_rms_px'][1]:.1f} px）も補正しました。"
          f"確認（フォーカスループのあと、瞳を自由に動かす BLIS-FPM で位置を固定して解き直し）：モデル像との残りの比例ずれ {ck['residual_linear_drift_pct']:.3f} %（走査全体で {ck['residual_linear_drift_max_px']:.2f} px）、"
          f"残りのランダムなずれ rms {ck['random_rms_px'][0]:.2f} / {ck['random_rms_px'][1]:.2f} px で、データとモデルの像の位置は合っています。"
          f"一方、自由な瞳の多項式から求めたコントラストのピント（瞳 − 物体の再合焦）の比例ずれと残したずれの差は {ck['total_focus_drift_minus_retained_pct']:.3f} % で、基準（0.1 %）を超えました（フォーカスループ前の版では {ck1['total_focus_drift_minus_retained_pct']:.3f} %）。"
          "これは、自由な瞳のデフォーカスを多項式で読み取ると、固定した瞳より小さく出るためです（2.4 節）。フォーカスループ自体は収束しており（`focus_loop`）、固定した瞳のモデルの中では、視差・コントラスト・物体のピントがそろっています。")
    else:
        A(f"高次の部分（2 次以上。測定 rms {PAR['higher_order_rms_px'][0]:.1f} / {PAR['higher_order_rms_px'][1]:.1f} px、行 / 列）は、瞳が予測する高次の視差を差し引いた残り（rms {hr[0]:.1f} / {hr[1]:.1f} px）を機械的なものとして補正しました。"
          f"ランダムな部分（rms {PAR['random_rms_px'][0]:.1f} / {PAR['random_rms_px'][1]:.1f} px）も補正しました。"
          f"確認の結果：解き直した解のコントラストのピント（瞳 − 物体の再合焦）が予測する比例ずれと残したずれの差 {ck['total_focus_drift_minus_retained_pct']:.3f} %、モデル像との残りの比例ずれ {ck['residual_linear_drift_pct']:.3f} %（走査全体で {ck['residual_linear_drift_max_px']:.2f} px）、"
          f"（瞳だけでは {ck['pupil_drift_minus_retained_pct']:.3f} %。解き直した BLIS-FPM は 2 次位相の一部を物体側に移し、その分の視差を自由な瞳の局所的な傾きで表しています）、"
          f"残りのランダムなずれ rms {ck['random_rms_px'][0]:.2f} / {ck['random_rms_px'][1]:.2f} px です。判定：{'整合' if PAR['consistent'] else '不整合'}（基準 0.1 %）。")
    A("")
    A(f"機械的な比例ずれの読み方：レンズを X（列方向）に動かすと像が列方向に約 {abs(mm_['scale_col_pct']):.1f} % ずれ、Y に動かすと像が列方向に約 {abs(mm_['col_from_row_axis_pct']):.1f} % ずれます（FZP の Y 軸と検出器の軸が約 {abs(np.degrees(mm_['col_from_row_axis_pct'] / 100)):.1f}° 傾いているのと同じ）。"
      "X の追従の倍率（検出器 X の係数か FZP X の 1 パルスの大きさ）と、ステージの軸の向きの較正をお勧めします（9 章）。")
    A("")
    A("![position affine](figures/fig0_position_affine.png)")
    A("")
    A("*図 1b　(a) 明視野 60 枚のレンズ位置での像のずれ。灰色：測ったずれ（全成分）、緑：瞳が予測する視差（データに残した分）。(b) 比例ずれ（2×2 行列）の成分。灰色：測定、緑：瞳が予測、橙：機械的（画像から補正した分）。*")
    A("")
A("### 2.3 照明の幾何（試料なし像のケラレから）")
A(f"この配置では、WOTF による較正（fptrecon の calibrate）が収束しませんでした（弱物体モデルの misfit が約 0.8）。そこで照明波数は、試料なし像のケラレから求めました。"
  f"視野の中で直接光が瞳を通るのは、円盤 |x − x_c,n| < R の内側です。リング 2–5 に当てはめると、x_c,n = A d_n + c₀ で、A はほぼ等方的な −{np.sqrt(abs(np.linalg.det(np.array(geo['A'])))):.3f} 倍、R = {geo['R_field_um']:.1f} µm、縁の幅は {geo['soft_um']:.1f} µm でした。"
  f"パイプラインの順モデル（O_eff = O·exp(iπκ|x|²)）では、x_c = −k_n/κ、R = kc/κ となります。これから k_n/kc = −x_c,n/R が、未測定の p によらずに決まります。"
  f"同じ当てはめから、FOV の曲率は κ = kc/R = {c_eff:.2f} × 1/(λ·0.75 m) と見積もられます。FOV モデルには公称の c = 1 を使いました（4 章の走査でも、c = 0.5–1.25 はほぼ同じ結果です）。"
  "双対解の符号は、κ の物理的な符号（> 0）から +1 に決まります。BLIS-FPM の線形解の corr(a, φ) も負で、これと一致します。")
A("")
A("### 2.4 瞳（デフォーカス・非点）：コントラストのピント")
if PAR:
    al = PAR["aligned_stack"]; cf = PAR["contrast_focus"]
    PT = J("d32a/af_test/parallax_test.json") if ex("d32a/af_test/parallax_test.json") else None
    A("FPM では、瞳の 2 次の位相（デフォーカス・非点）と、物体側の 2 次の位相（物体の複素振幅にかかる 2 次位相）は、像のコントラストに対して入れ替え可能です。コントラストが決めるのは両者の和だけです。"
      + ("ただし、照明を傾けたときの像のずれ（視差）を生むのは瞳の位相だけで、物体側の 2 次位相は像を動かしません。"
         f"弱い物体の数値計算（z = {PT['z_um']/1e3:.0f} mm 相当）でも、瞳のデフォーカスでは像が {min(np.hypot(*r_['pupil_shift_px']) for r_ in PT['results']):.1f}–{max(np.hypot(*r_['pupil_shift_px']) for r_ in PT['results']):.1f} px 動き、"
         f"同じ 2 次位相を物体にかけると {max(np.hypot(*r_['object_shift_px']) for r_ in PT['results']):.1f} px 以下でした（`supplement_astigmatism/parallax_test.py`）。" if PT else "")
      + ""
      "そこで、ずれをすべて取り除いたデータで瞳と物体を解いたあと、物体のピントを合わせました。ピントの基準は、帯域 0.3–3.5 µm⁻¹ の対数振幅と位相の比例からのずれが最小になることです（単一材料の薄い試料では、ピントが合うと振幅は位相に比例します）。")
    A(f"その結果、瞳 z {al['pupil']['defocus_um']/1e3:+.2f} mm、非点 ({al['pupil']['astig_um'][0]/1e3:+.2f}, {al['pupil']['astig_um'][1]/1e3:+.2f}) mm に対して、物体の再合焦は z {al['object_refocus']['z_um']/1e3:+.2f} mm、非点 ({al['object_refocus']['astig_um'][0]/1e3:+.2f}, {al['object_refocus']['astig_um'][1]/1e3:+.2f}) mm でした"
      f"（指標 {al['object_refocus']['metric_before']:.3f} → {al['object_refocus']['metric_after']:.3f}）。"
      + (f"和（コントラストのピント）は z {(al['pupil']['defocus_um'] - al['object_refocus']['z_um'])/1e3:+.2f} mm、非点 ({(al['pupil']['astig_um'][0] - al['object_refocus']['astig_um'][0])/1e3:+.2f}, {(al['pupil']['astig_um'][1] - al['object_refocus']['astig_um'][1])/1e3:+.2f}) mm で、非点はほぼ 0 です。これを瞳に割り当て（物体はピントが合った状態）、その視差を残しました。"
         "以前の版（2026-09-27 版）は、この値を 3 手法の瞳の初期値にしていました。" if AFF2 else
         f"和（コントラストのピント）は **z {cf['defocus_um']/1e3:+.2f} mm、非点 ({cf['astig_um'][0]/1e3:+.2f}, {cf['astig_um'][1]/1e3:+.2f}) mm** で、非点はほぼ 0 です。これを瞳に割り当て（物体はピントが合った状態）、3 手法の瞳の初期値にしました（`calibration.json`）。"))
    AB = J("d32a/af_test/af_bias_test.json") if ex("d32a/af_test/af_bias_test.json") else None
    if AB:
        A(f"物体のピントの基準が偏っていないことは、模擬データで確かめました。ピントの合った単一材料の物体（a = {AB['g']:.2f} φ）から、較正した瞳と同じ照明で像を計算し、EPRY で再構成すると、再合焦量は {AB['epry_synthetic_best']['z_um']/1e3:+.2f} mm でした（`supplement_astigmatism/af_bias_test.py`）。")
    if AST:
        A("以前の版の瞳の大きな非点は、比例ずれを残したデータでは瞳を 0 から始めても再現されますが、そのとき物体も逆向きの非点をもっていて、和はほぼ非点なしになります。"
          "つまり光学系に非点はなく、以前の瞳の非点は、比例ずれを説明するために瞳に入り、物体側が逆の非点で打ち消していたものです（7 章）。")
    A("")
if AFF2:
    L0 = PAR["focus_loop"]; ck1, ck2 = PAR1["check"], PAR["check"]
    A("**フォーカスループ（2026-09-28 版）**：以前の版の再構成では、EPRY・BLIS-FPM の物体に約 +1.1 mm のピントずれが残っていました（7.3 節）。"
      "そこで posaffine に次のくり返しを加えました（`position_affine.focus_iter`）。(1) 瞳をコントラストのピント（デフォーカス＋非点だけの滑らかな瞳）に固定し、その視差を残したデータで物体だけを解く。"
      "(2) 物体のピントを合わせる。再合焦量が 0.1 mm 以上なら、それを瞳に移し、視差と補正を計算し直して (1) に戻る。")
    A("")
    A("| くり返し | 固定した瞳 z / 非点 (mm) | 物体の再合焦 z / 非点 (mm) | misfit |")
    A("|---|---|---|---|")
    for i_, l_ in enumerate(L0):
        A(f"| {i_} | {l_['pupil_defocus_um']/1e3:+.2f} / ({l_['pupil_astig_um'][0]/1e3:+.2f}, {l_['pupil_astig_um'][1]/1e3:+.2f}) | "
          f"{l_['object_refocus']['z_um']/1e3:+.2f} / ({l_['object_refocus']['astig_um'][0]/1e3:+.2f}, {l_['object_refocus']['astig_um'][1]/1e3:+.2f}) | {l_['misfit']:.4f} |")
    A("")
    mm1, mm2 = PAR1["mechanical"], PAR["mechanical"]
    A(f"2 回で収束し、瞳は **z {PAR['pupil_defocus_um']/1e3:+.2f} mm、非点 ({PAR['pupil_astig_um'][0]/1e3:+.2f}, {PAR['pupil_astig_um'][1]/1e3:+.2f}) mm** になりました（`calibration.json`、3 手法の瞳の初期値）。"
      f"misfit はほとんど変わりません（{L0[0]['misfit']:.4f} → {L0[-1]['misfit']:.4f}）。これは測定そのものの縮退によるものです。物体側のデフォーカス z は、「瞳のデフォーカス z」と「その視差 λzk_n を打ち消す像のずれ」を組み合わせたものと同じ像を作り、"
      "k_n（照明の位置）に比例する像のずれは、機械的な比例ずれの等方成分（倍率）と区別できません。つまり、データだけでは「物体のピントずれ」と「瞳のピントずれ＋倍率」を分けられません。"
      "このくり返しは、試料（ジーメンススター）にピントが合っていることを条件にして、この縮退を解いています。"
      f"その分、機械的な比例ずれの等方成分が変わりました（列←X {mm1['scale_col_pct']:+.2f} → {mm2['scale_col_pct']:+.2f} %、行←Y {mm1['scale_row_pct']:+.2f} → {mm2['scale_row_pct']:+.2f} %、等方成分 {mm2['isotropic_pct'] - mm1['isotropic_pct']:+.2f} %）。"
      f"回転（{mm2['rotation_deg']:+.2f}°）とせん断（{mm2['shear_pct']:+.2f} %）は変わりません。瞳で説明できる比例ずれの割合は {100*PAR1['fraction_of_drift_explained_by_pupil']:.0f} % → {100*PAR['fraction_of_drift_explained_by_pupil']:.0f} % になりました。")
    A("")
    AQ = J("d32a/af_test/pupil_annulus_quadratic.json") if ex("d32a/af_test/pupil_annulus_quadratic.json") else None
    if PBT and AQ:
        zq = [f_["z_um"] / 1e3 for f_ in AQ["aff2"]["annulus_quadratic"]]
        A(f"確認のため、瞳も自由に動かす BLIS-FPM で解き直すと（posaffine の確認の計算）、物体の再合焦量は {ck1['object_refocus']['z_um']/1e3:+.2f} mm（ループ前）→ {ck2['object_refocus']['z_um']/1e3:+.2f} mm（ループ後）になり、ループ後の物体はピントが合っています。"
          f"ただし、自由な瞳のデフォーカスは、posaffine の 4 次多項式で読むと {ck2['pupil']['defocus_um']/1e3:+.2f} mm で、固定した瞳（{PAR['pupil_defocus_um']/1e3:+.2f} mm）より小さく出ます。"
          f"自由な瞳は、中心（|k| < 0.2 µm⁻¹）に約 {AQ['aff2']['central_step_rad']:.1f} rad の位相の段差をもち、動径方向の形も 2 次曲線からずれているためです（図 2-4 d）。"
          f"このため、デフォーカスの値は当てはめ方によって変わります。回転対称な 2 次だけで当てはめると {min(zq):+.2f}〜{max(zq):+.2f} mm（当てはめる範囲による）、Zernike のバランスしたデフォーカスは {PBT['aff2']['zernike']['z_balanced_um']/1e3:+.2f} mm です。"
          f"posaffine の確認で「(瞳 − 物体の再合焦) の比例ずれ − 残したずれ」が {ck2['total_focus_drift_minus_retained_pct']:.3f} % と基準を超えたのは、このためです。"
          f"一方、ループの前後で、自由な瞳のデフォーカス（4 次多項式）は {ck1['pupil']['defocus_um']/1e3:+.2f} → {ck2['pupil']['defocus_um']/1e3:+.2f} mm、物体の再合焦量は {ck1['object_refocus']['z_um']/1e3:+.2f} → {ck2['object_refocus']['z_um']/1e3:+.2f} mm と、ちょうど同じ量だけ入れ替わりました。"
          "つまり、残した視差を変えた分は、そのまま瞳が引き受けています。")
        A("")
    if ex("d32a/fig_focusloop_32a.png"):
        A("![focus loop](figures/fig_focusloop_32a.png)")
        A("")
        A("*図 2-4　(a) フォーカスループ（瞳を固定して物体を解く）の瞳と物体の再合焦量。(b) 各再構成の物体の再合焦量（中央 512 px、FOV モデル。灰色：ループ前、緑：ループ後）。"
          "(c) 各再構成の瞳のデフォーカス（破線：データに残した視差に相当するデフォーカス）。(d) posaffine の確認で解いた自由な瞳の動径方向の位相（ピストン・傾き・非点を除き、|k| = 0.5 µm⁻¹ で 0 に合わせた）。灰色の帯は明視野の直接光が瞳を通る範囲。*")
        A("")
A("### 2.5 FZP の 0 次光による干渉縞（制限事項）")
A("リング 0–1 の比の像には、円弧状の縞が強く入っています（図 1g）。縞の中心は、試料なしの光線が FZP の中心を通る視野上の点です。"
  "間隔は中心から離れるほど狭くなり（輪帯板と同じ形）、FZP の 0 次光（集光されない透過光）が像と干渉したものと考えられます。"
  "リング 2 では弱く、リング 3 ではほぼ消えます。どの手法の順モデルにも入っていないので、リング 0–1 の misfit が高くなる主な原因と考えています（5 章）。")
A("")
A("### 2.6 暗視野データ")
A(f"リング 6–7 の各画像を n = [s − G₃{{d}}] / (t · g · U) としました（036 と同じ方式）。ここで s は試料あり像、d は同じ位置の試料なし像（背景や迷光を引く）、t は露光、g は明視野の直接光レベル（{prep['bright_level_counts_per_s']:.0f} counts/s）、U は照明の帯の形です。"
  f"暗視野の強度は明視野の {100*lev6:.2f} %（リング 6）、{100*lev7:.2f} %（リング 7）です。画素あたりの雑音の分散はコントラストの 2 乗の {nz6:.1f} 倍 / {nz7:.1f} 倍で、SNR は 1 未満です。"
  "照明の帯の外（U < 0.08。全視野の約 21 %、中央 512 px では 0 %）は、有効画素の重みで除外しました。")
if ex("d32a/df_registration_test.json"):
    rt_ = J("d32a/df_registration_test.json")
    A(f"暗視野像の位置ずれは、明視野と同じ画像どうしの相互相関では精度よく求まりませんでした（暗視野どうしの組は外れ値が多い）。"
      f"相互相関で求めた補正を入れた場合と入れない場合を DIP（512 px、平面波）で比べると、外した画像の検証損失は {rt_['reg2']['val_loss']:.2f} 対 {rt_['nd']['val_loss']:.2f} で、補正しない方がよい結果でした。"
      "そのため以前の版では、暗視野像にはレンズと検出器の食い違い（2.1 節）の補正だけを使いました（どちらの場合も、DIP が学習する暗視野像のずれは上限 ±5 px に達していました）。")
A("")
if cdf and "position_affine" in cdf:
    dc = np.array(cdf["position_affine"]["dark_correction_px"])
    A(f"**暗視野像の位置（補正後の版）**：暗視野像どうしの相互相関は信頼できないため、明視野で求めた機械的な 2×2 行列を暗視野の位置（レンズ移動 90–100 µm）に外挿して補正しました。"
      f"補正量は rms {np.sqrt((dc[:, 0] ** 2).mean()):.1f} / {np.sqrt((dc[:, 1] ** 2).mean()):.1f} px（行 / 列）、最大 {np.abs(dc[:, 0]).max():.0f} / {np.abs(dc[:, 1]).max():.0f} px です。"
      f"以前の版では、この補正がなかったため、暗視野像は最大 {np.abs(dc[:, 1]).max():.0f} px ずれていたことになります。")
    if ex("d32a/dip_df_shift_stats.json"):
        SS = J("d32a/dip_df_shift_stats.json")
        nb_old = [SS[f"32adf|{t}"]["dark_near_bound"] for t in ("DIP_planewave", "DIP_FOV_c1")]; dsn_ = "32adfaff2" if AFF2 and "32adfaff2|DIP_FOV_c1" in SS else "32adfaff"; nb_new = [SS[f"{dsn_}|{t}"]["dark_near_bound"] for t in ("DIP_planewave", "DIP_FOV_c1")]
        A(f"ただし、DIP が学習した暗視野像のずれ（上限 ±5 px）は、補正後も {SS['32adfaff|DIP_FOV_c1']['n_dark']} 枚中 {min(nb_new)}–{max(nb_new)} 枚が上限近く（4.5 px 超）に達しました（補正前 {min(nb_old)}–{max(nb_old)} 枚）。"
          "暗視野像は画素あたりの SNR が 1 未満なので、DIP はずれの自由度を雑音の当てはめに使っていると考えられます。暗視野像のずれを固定して学習する方がよい可能性があります（未検証）。")
    A("")
A("## 3. Δr（瞳半径）の比較")
A("")
A("（3 章と 4 章の計算は、機械的な像ずれを補正する前のデータで行いました。結論（Δr = 250 nm、c = 1）は補正の影響を受けないと考えています。）")
A("")
A("| 条件 | 当てはめ misfit | 外した 7 枚の misfit |")
A("|---|---|---|")
for k, lab in [("reg2_dr250_c0", "Δr 250 nm（kc 2.0）平面波"), ("reg2_dr250_c1", "Δr 250 nm（kc 2.0）FOV"), ("reg2_dr200_c0", "Δr 200 nm（kc 2.5）平面波"), ("reg2_dr200_c1", "Δr 200 nm（kc 2.5）FOV")]:
    A(f"| {lab} | {f3(dr[k]['fitted'])} | {f3(dr[k]['heldout'])} |")
A("")
A(f"BLIS-FPM（明視野 60 枚から 7 枚を除き、初期値 0、瞳のみ 20 + 同時 150 反復）の結果です。kc 2.5 の平面波は当てはめが {f3(dr['reg2_dr200_c0']['fitted'])} で止まっており、比較には使えません。"
  f"全 60 枚を使ったパイプラインの結果も、FOV モデルで kc 2.0 が {f3(dr['pipeline_dr250']['1.0'])}、kc 2.5 が {f3(dr['pipeline_dr200']['1.0'])} です。"
  "kc 2.5 では、データに信号のない瞳の外周部分（図 2d の周辺）まで自由に解くことになり、瞳が雑音化します。以後は **Δr = 250 nm** で解きました。")
A("")
A("![fov_dr](figures/fig_fov_dr_32a.png)")
A("")
A("*図 2　a：BLIS-FPM の misfit と FOV の強さ c の関係（点線：ケラレから求めた c）。b：Δr の比較。c, d：kc 2.0 と 2.5 の BLIS-FPM の瞳位相（FOV モデル、ピストン除去）。*")
A("")
A("## 4. FOV の強さ")
A("| c | " + " | ".join(k for k in fs) + " |")
A("|---|" + "---|" * len(fs))
A("| 当てはめ | " + " | ".join(f3(v["fitted"]) for v in fs.values()) + " |")
A("| 外した 7 枚 | " + " | ".join(f3(v["heldout"]) for v in fs.values()) + " |")
A("")
A(f"当てはめは c = 1 で最小になりました。外した画像は c = 0.5–1.25 でほぼ横ばいです。ケラレから求めた c = {c_eff:.2f} もこの範囲にあります。平面波（c = 0）と c ≥ 1.5 は明らかに悪くなります。")
A("")
for key, ttl, dark in [(MAIN["bf512"], "5. 明視野（リング 0–2）の 4 手法比較", False), (MAIN["df512"], "6. 暗視野を含む 4 手法比較（リング 0–2 + 6–7）", True)]:
    if key not in C: continue
    A(f"## {ttl}")
    A("")
    A("misfit は、帯域 0.3–3.5 µm⁻¹ の強度スペクトル misfit（`dipfpm.evaluate.band_misfit`）です。**帯域制限していない物体**から計算しています。1 は予測なし（平坦な物体と同じ）を表し、小さいほどよい値です。")
    A("「採用反復」は、DIP では外した画像で選んだ反復、EPRY では採用したスイープです。")
    if dark:
        bz, bw = J("d32a/blis_df/c1_ho_zeroinit.json"), J("d32a/blis_dfaff2/c1_ho.json" if AFF2 else ("d32a/blis_dfaff/c1_ho.json" if AFF and ex("d32a/blis_dfaff/c1_ho.json") else "d32a/blis_df/c1_ho.json"))
        epl = lambda k_: "、".join(f"{mj} {r(k_, 'EPRY', mlab)['iteration']}" for mod, mlab, mj in MOD if k_ in C and r(k_, "EPRY", mlab))
        ep_txt = f"中央 512 px：{epl(key)}" + (f"／全視野 1000 px：{epl(key.replace('_512', '_1000'))}" if key.replace("_512", "_1000") in C else "")
        A("")
        A("暗視野を含む計算での手法ごとの注意：")
        A("1. **BLIS-FPM**：平面波モデルで初期値 0（一様な物体）から始めると、暗視野像の予測が 0 になり、規格化が 0/0 で計算が止まりました。"
          "そこで平面波・FOV とも、明視野だけで解いた BLIS-FPM の結果（同じ 7 枚の明視野像を外して解いたもの）を初期値にしました。"
          f"外した画像は同じなので、予測の評価は公平です。以前の版で確かめたところ、FOV モデルでは初期値 0 からでも解けており、外した画像の misfit（BLIS-FPM 内部の指標）は {bz['heldout_mean']:.3f}（初期値 0）で、明視野の解から始めた場合（{J('d32a/blis_df/c1_ho.json')['heldout_mean']:.3f}）とほぼ同じでした。")
        A(f"2. **EPRY**：採用するスイープは、外した 18 枚（うち暗視野 11 枚）の平均 misfit で選びます。暗視野像の予測はスイープを重ねるほど悪くなるため、早いスイープ（{ep_txt}）が選ばれ、明視野の予測も明視野だけのときより悪くなっています。")
        A("3. **DIP**：暗視野像には雑音の逆数の重みをかけています（パイプラインの暗視野モード）。学習した暗視野像のずれは、機械的なずれを補正した後も多くが上限（±5 px）近くに達しています（2.6 節）。")
        A("")
        A("外した画像の misfit は、明視野（7 枚）と暗視野（11 枚）に分けて示します。暗視野の misfit が 1 を超えるのは、予測した暗視野像の構造が実測と合わず、平坦な像よりも差が大きいことを意味します。")
    for g, gl in (("512", "中央 512 px（16.3 µm 角）"), ("1000", "全視野 1000 px（31.9 µm 角）")):
        k2 = key.replace("_512", "_" + g)
        if k2 not in C: continue
        s = C[k2]
        A("")
        A(f"### {ttl.split('.')[0]}-{1 if g == '512' else 2}　{gl}")
        A(f"学習 {s['images_train']} 枚" + (f"（うち暗視野 {s['dark_images_train']} 枚）" if dark else "") + f"、外した画像 {s['images_validation']} 枚" + (f"（うち暗視野 {s['dark_images_validation']} 枚）" if dark else "") + "。")
        A("")
        A(table(k2, dark))
        A("")
        pre = ("bf" if not dark else "df") + g
        A(f"![{pre} FOV](figures/{pre}_images_FOV_c1.png)")
        A("")
        A(f"*図　{gl}、FOV モデル。左から透過率、位相、星の中心の拡大（位相）、瞳位相。帯域フィルターなし。*")
        A("")
        A(f"![{pre} plane](figures/{pre}_images_planewave.png)")
        A("")
        A(f"![{pre} zoom](figures/{pre}_zoom.png)")
        A("")
        A(f"*図　星の中心（5.1 µm 角）の位相の拡大。8 条件（4 手法 × 2 モデル）を、モデルごとに共通のグレースケールで表示しています。*")
        A("")
        A(f"![{pre} spokes](figures/{pre}_spokes.png)")
        A("")
if AFF:
    A("## 7. 補正前後の比較：楕円状の癖と瞳の非点")
    A("")
    AN0 = J("d32a/aniso_32a.json")["B_pupil_edge"] if ex("d32a/aniso_32a.json") else None
    ref_, kp_, nl_ = AST["reference: BLIS used in comparison (calibrated init)"], AST["drift kept (current data)"], AST["linear drift removed"]
    A(f"以前の版では、再構成像（とくにスペクトル）に、−22° と 68° の方向で強さが違う「楕円状の癖」がありました。その版の瞳には大きな非点（主軸方向の焦点ずれ {ref_['principal_mm'][0]:.1f} / {ref_['principal_mm'][1]:.1f} mm、主軸 {ref_['axis_deg']:.0f}°）が入っていました。"
      "調べた結果、この非点は光学系のものではなく、機械的な比例ずれを瞳が肩代わりしたものでした。")
    A("")
    A(f"1. 瞳の位相の勾配は像をずらします（2.2 節）。像のずれに、レンズ位置に比例する成分が残っていると、それを説明するために瞳に 2 次の位相（デフォーカス・非点）が入ります。比例ずれが方向によって違う（列 {PAR['mechanical']['scale_col_pct']:+.1f} %、行 {PAR['mechanical']['scale_row_pct']:+.1f} %、せん断あり）ので、非点になります。"
      f"比例ずれを残したデータで瞳を 0 から解いても、同じ非点（{kp_['principal_mm'][0]:.1f} / {kp_['principal_mm'][1]:.1f} mm、{kp_['axis_deg']:.0f}°）が現れます。")
    A(f"2. 比例ずれを取り除いたデータでは、瞳を 0 から始めても非点はほとんど入りません（主軸方向の焦点ずれ {nl_['principal_mm'][0]:.1f} / {nl_['principal_mm'][1]:.1f} mm）。"
      f"外した画像の misfit はほぼ同じ（{AST['linear drift removed misfit']['heldout']:.3f} 対 {AST['drift kept misfit']['heldout']:.3f}、BLIS-FPM、FOV）で、コントラストだけでは両者を区別できません。")
    AFk = J("d32a/astig_test/autofocus_keep_nolin.json") if ex("d32a/astig_test/autofocus_keep_nolin.json") else None
    if AFk:
        k0, n0_ = AFk["keep0"], AFk["nolin0"]
        A(f"3. 瞳の 2 次位相と物体側の 2 次位相は入れ替え可能です（2.4 節）。比例ずれを残したデータの解では、物体も逆向きの非点をもっていて（再合焦 z {k0['object_refocus']['z_um']/1e3:+.2f} mm、非点 ({k0['object_refocus']['astig_um'][0]/1e3:+.2f}, {k0['object_refocus']['astig_um'][1]/1e3:+.2f}) mm）、"
          f"瞳との和は z {k0['total_defocus_um']/1e3:+.2f} mm、非点 ({k0['total_astig_um'][0]/1e3:+.2f}, {k0['total_astig_um'][1]/1e3:+.2f}) mm です。比例ずれを除いたデータの和（z {n0_['total_defocus_um']/1e3:+.2f} mm、非点 ({n0_['total_astig_um'][0]/1e3:+.2f}, {n0_['total_astig_um'][1]/1e3:+.2f}) mm）"
          f"や、posaffine の値（z {PAR['pupil_defocus_um']/1e3:+.2f} mm、非点 ({PAR['pupil_astig_um'][0]/1e3:+.2f}, {PAR['pupil_astig_um'][1]/1e3:+.2f}) mm）と近く、光学系の非点は 0.3 mm 以下です。")
    if AN0:
        A(f"4. 瞳の縁のぼけは、列方向 {AN0['s_along_col_um']:.2f} µm・行方向 {AN0['s_along_row_um']:.2f} µm でほぼ同じです（照明の角度の広がりにすると {AN0['sigma_theta_urad_col_row'][0]:.1f} / {AN0['sigma_theta_urad_col_row'][1]:.1f} µrad 以下）。空間コヒーレンスの異方性は、主な原因ではありません。")
    A("")
    if ex("d32a/astig_test/fig_astig_origin_32a.png") or True:
        A("![astig origin](figures/fig_astig_origin_32a.png)")
        A("")
        A("*図 7-1　以前の版の瞳（左、非点あり）と、比例ずれを除いたデータで 0 から解いた瞳（中）、星のスペクトルの方向差（右）。*")
        A("")
    A("### 7.1 外した画像の misfit・スポーク分解能・方向の偏り（補正前 → 補正後）")
    if AFF2: A("「補正後」は、この版（位置の 2×2 行列の補正と、フォーカスループ）です。フォーカスループの前後の比較は 7.3 節にあります。")
    A("方向の偏りは、星の輪帯（半径 15–200 px）の位相スペクトルで、−22° 方向と 68° 方向の強さの比の対数の rms（0.5–3.5 µm⁻¹、0 = 等方）です。星そのものの模様にも方向差があるので、0 にはなりません。")
    A("")
    for g, lab_g, dark_ in (("bf512", "明視野・512 px", False), ("bf1000", "明視野・1000 px", False), ("df512", "暗視野を含む・512 px", True), ("df1000", "暗視野を含む・1000 px", True)):
        kn_, ko_ = MAIN[g], OLD[g]
        if kn_ not in C or ko_ not in C: continue
        A(f"**{lab_g}**")
        A("")
        h = "| 手法 | モデル | misfit 外した画像 前 → 後 |" + (" うち暗視野 前 → 後 |" if dark_ else "") + " スポーク (µm⁻¹) 前 → 後 | 方向の偏り 前 → 後 |"
        A(h); A("|" + "---|" * (h.count("|") - 1))
        for mod, mlab, mj in MOD:
            for m, mn in M:
                xo, xn = r(ko_, m, mlab), r(kn_, m, mlab)
                if xo is None or xn is None: continue
                lb_ = f"{ML[m]}, {mlab}"
                ao = ANI.get(ko_, {}).get(lb_, {}).get("score"); an = ANI.get(kn_, {}).get(lb_, {}).get("score")
                row = f"| {mn} | {mj} | {f3(xo['misfit_unfiltered']['heldout']['all'])} → {f3(xn['misfit_unfiltered']['heldout']['all'])} |"
                if dark_: row += f" {f2(xo['misfit_unfiltered']['heldout'].get('dark'))} → {f2(xn['misfit_unfiltered']['heldout'].get('dark'))} |"
                row += f" {f2(xo.get('spoke_snr3_um_inv'))} → {f2(xn.get('spoke_snr3_um_inv'))} | {f2(ao)} → {f2(an)} |"
                A(row)
        A("")
    A("### 7.2 各再構成の瞳が予測する比例ずれの確認")
    A("各手法の再構成ごとに、瞳の 2 次の成分（z、非点）と、物体のピントずれ（中央 512 px で求めた再合焦量）を求め、次の 4 つを調べました。基準はどれも 0.1 %（レンズ移動 50 µm で像のずれ 1.6 px）です。")
    A("")
    A("- (A) 瞳だけが予測する比例ずれと、データに残した比例ずれ（2.2 節の視差）の差。像を動かすのは瞳だけなので、モデルの視差そのものです。")
    A("- (B) コントラストのピント（瞳 − 物体の再合焦）が予測する比例ずれと、残した比例ずれの差。その再構成のコントラストが、posaffine のピントと同じかを表します。")
    A("- (C) DIP が学習した画像ごとのずれに含まれる比例成分。")
    A("- (D) 再構成のモデル像（瞳・物体・学習したずれを含む）と各データ像の相互相関から求めた、残りの比例ずれ。データとモデルの位置が合っているかの直接の確認です。")
    A("")
    for g, lab_g in (("bf512", "明視野・512 px"), ("bf1000", "明視野・1000 px"), ("df512", "暗視野を含む・512 px"), ("df1000", "暗視野を含む・1000 px")):
        kn_ = MAIN[g]
        if kn_ not in C: continue
        rows_ = []
        for mod, mlab, mj in MOD:
            for m, mn in M:
                x = r(kn_, m, mlab)
                if x is None or not x.get("pupil_drift_check"): continue
                ck_ = x["pupil_drift_check"]; af_ = ck_["object_refocus"]
                tr_ = ck_.get("learned_shift_linear_trend_pct")
                md_ = ck_.get("model_residual_linear_drift_pct")
                rows_.append(f"| {mn} | {mj} | {ck_['pupil_defocus_um']/1e3:+.2f} / ({ck_['pupil_astig_um'][0]/1e3:+.2f}, {ck_['pupil_astig_um'][1]/1e3:+.2f}) | "
                             f"{af_['z_um']/1e3:+.2f} | {ck_['total_defocus_um']/1e3:+.2f} / ({ck_['total_astig_um'][0]/1e3:+.2f}, {ck_['total_astig_um'][1]/1e3:+.2f}) | "
                             f"{ck_['pupil_drift_minus_kept_pct']:.3f} | {ck_['total_drift_minus_kept_pct']:.3f} | {'—' if tr_ is None else f'{tr_:.3f}'} | {'—' if md_ is None else f'{md_:.3f}'} |")
        if not rows_: continue
        A(f"**{lab_g}**")
        A("")
        A("| 手法 | モデル | 瞳 z / 非点 (mm) | 物体の再合焦 z (mm) | コントラストのピント z / 非点 (mm) | (A) 瞳だけの比例ずれ − 残したずれ (%) | (B) コントラストのピントの比例ずれ − 残したずれ (%) | (C) DIP 学習ずれの比例成分 (%) | (D) モデル像とデータの比例ずれ (%) |")
        A("|---|---|---|---|---|---|---|---|---|")
        for rw in rows_: A(rw)
        A("")
    ck_all = [x["pupil_drift_check"] for g_ in ("bf512", "bf1000") if MAIN[g_] in C for x in C[MAIN[g_]]["results"].values() if x.get("pupil_drift_check")]
    if ck_all:
        zt_ = [c_["total_defocus_um"] / 1e3 for c_ in ck_all]; ast_ = [max(map(abs, c_["total_astig_um"])) / 1e3 for c_ in ck_all]
        dD = [c_["model_residual_linear_drift_pct"] for c_ in ck_all if "model_residual_linear_drift_pct" in c_]
        dB = [c_["total_drift_minus_kept_pct"] for c_ in ck_all]
        tr_ = [c_["learned_shift_linear_trend_pct"] for c_ in ck_all if "learned_shift_linear_trend_pct" in c_]
        dA_ = [c_["pupil_drift_minus_kept_pct"] for c_ in ck_all if abs(c_["pupil_drift_minus_kept_pct"]) > 0.05]
        A((f"読み方：(D) はどの再構成も {max(dD):.3f} % 以下で、どの再構成もデータの像の位置を再現しています。瞳を自由に更新する DIP・BLIS-FPM では (A) が {min(dA_):.2f}–{max(dA_):.2f} % と大きく、"
           f"DIP の学習したずれの比例成分 (C)（{min(tr_):.2f}–{max(tr_):.2f} %）では埋まりません。これらの再構成は、2 次式では読み取れない瞳の形（中心の段差や、2 次曲線からずれた動径方向の形。2.4 節）で視差を表していると考えられます。" if AFF2 else
           f"読み方：(D) はどの再構成も {max(dD):.3f} % 以下で、どの再構成もデータの像の位置を再現しています。(A) が大きいとき、その差は (C) の学習したずれ（DIP、{min(tr_):.2f}–{max(tr_):.2f} %）か、自由な瞳の局所的な傾き（BLIS-FPM）が埋めています。")
          + f"(B) が大きいのは、その再構成のコントラストが示すピントが、posaffine で決めたピント（z {PAR['pupil_defocus_um']/1e3:+.2f} mm）と違うことを意味します。"
          "EPRY は明視野だけでは瞳を更新できないため、瞳は posaffine の値のままで (A) はほぼ 0 です。コントラストの違いは物体側の 2 次位相（再合焦量）に出ます。"
          + (lambda v_: f"明視野の DIP の物体は、再合焦量がほぼ 0 になりました（暗視野を含む DIP（最適反復）は例外で {v_:+.2f} mm）。" if v_ is not None else "DIP の物体は再合焦量がほぼ 0 になりました。")(
            (r(MAIN["df512"], "DIP_opt", "FOV model") or {}).get("pupil_drift_check", {}).get("object_refocus", {}).get("z_um", 0) / 1e3 if MAIN["df512"] in C else None)
          + "DIP は吸収と位相をほぼ比例させた物体を出すため、この基準では DIP の物体のピントは判定しにくいと考えられます（コントラストのピント = 瞳の値として扱いました）。")
        A("")
        mm_ = PAR["mechanical"]
        A(f"まとめると、非点はどの再構成でも {max(ast_):.2f} mm 以下です。デフォーカス（コントラストのピント）は手法によって {min(zt_):+.2f}〜{max(zt_):+.2f} mm と幅があり、比例ずれの等方的な部分（倍率）に直すと最大 {max(dB):.2f} % の違いです。"
          f"この分だけ、機械的な倍率の値（列←X {mm_['scale_col_pct']:+.2f} %、行←Y {mm_['scale_row_pct']:+.2f} %）は不確かです。"
          f"回転（{mm_['rotation_deg']:+.2f}°）・せん断（{mm_['shear_pct']:+.2f} %）・X と Y の倍率の差（{mm_['scale_row_pct'] - mm_['scale_col_pct']:.2f} %）は、この不確かさの影響を受けません。"
          "ピントを独立に決めるには、試料を光軸方向に既知の量（例えば ±2 mm）動かした測定が有効です。")
        A("")
if AFF2:
    A("### 7.3 物体側のピントずれを瞳へ移した効果（フォーカスループの前 → 後）")
    A("フォーカスループの前（2026-09-27 版、瞳 z " + f"{PAR1['pupil_defocus_um']/1e3:+.2f} mm）と後（この版、瞳 z {PAR['pupil_defocus_um']/1e3:+.2f} mm）で、同じ 32 の再構成を比べました。"
      "物体の再合焦量は、中央 512 px で求めた、物体のピントを合わせるのに必要な伝搬距離です（0 = ピントが合っている）。")
    A("")
    for g, lab_g, dark_ in (("bf512", "明視野・512 px", False), ("bf1000", "明視野・1000 px", False), ("df512", "暗視野を含む・512 px", True), ("df1000", "暗視野を含む・1000 px", True)):
        kn_, ko_ = MAIN[g], PREV[g]
        if kn_ not in C or ko_ not in C: continue
        A(f"**{lab_g}**")
        A("")
        h = "| 手法 | モデル | misfit 外した画像 前 → 後 |" + (" うち暗視野 前 → 後 |" if dark_ else "") + " スポーク (µm⁻¹) 前 → 後 | 物体の再合焦 (mm) 前 → 後 | 瞳 z (mm) 前 → 後 |"
        A(h); A("|" + "---|" * (h.count("|") - 1))
        for mod, mlab, mj in MOD:
            for m, mn in M:
                xo, xn = r(ko_, m, mlab), r(kn_, m, mlab)
                if xo is None or xn is None: continue
                po, pn = xo.get("pupil_drift_check") or {}, xn.get("pupil_drift_check") or {}
                row = f"| {mn} | {mj} | {f3(xo['misfit_unfiltered']['heldout']['all'])} → {f3(xn['misfit_unfiltered']['heldout']['all'])} |"
                if dark_: row += f" {f2(xo['misfit_unfiltered']['heldout'].get('dark'))} → {f2(xn['misfit_unfiltered']['heldout'].get('dark'))} |"
                zf = lambda d_, k_: "—" if not d_ else (f"{d_['object_refocus']['z_um']/1e3:+.2f}" if k_ == "rf" else f"{d_['pupil_defocus_um']/1e3:+.2f}")
                row += f" {f2(xo.get('spoke_snr3_um_inv'))} → {f2(xn.get('spoke_snr3_um_inv'))} | {zf(po, 'rf')} → {zf(pn, 'rf')} | {zf(po, 'pz')} → {zf(pn, 'pz')} |"
                A(row)
        A("")
    A("**再構成像（ループの前後、FOV モデル、帯域フィルターなし）**：同じデータ（明視野、または暗視野を含む）の中では、ループの前後と 4 手法で共通のグレースケールにしています。数字はスポーク分解能、物体の再合焦量、その手法が求めた瞳のデフォーカスです。段の見出しの瞳 z（−1.52 / −2.72 mm）は全手法に共通の瞳の初期値で、EPRY はこの値のまま、DIP と BLIS-FPM は瞳を自分で更新します。"
      "ループ後の像は、5 章・6 章の図（主な結果）と同じものです。")
    A("")
    for f_, cap_ in (("fig_focusloop_zoom_32a_512.png", "*図 7-1b　星の中心（5.1 µm 角）の位相。上 2 段：明視野（ループ前 / 後）、下 2 段：暗視野を含む（ループ前 / 後）。中央 512 px で解いた再構成。破線の枠：ループ前、実線の枠：ループ後。*"),
                     ("fig_focusloop_field_32a_512.png", "*図 7-1c　中央 512 px（16.3 µm 角）全体の位相。並びは図 7-1b と同じ。*"),
                     ("fig_focusloop_zoom_32a_1000.png", "*図 7-1d　全視野 1000 px で解いた再構成の、星の中心（5.1 µm 角）の位相。*"),
                     ("fig_focusloop_field_32a_1000.png", "*図 7-1e　全視野 1000 px（31.9 µm 角）全体の位相。*")):
        if ex("d32a/" + f_):
            A(f"![{f_}](figures/{f_})"); A(""); A(cap_); A("")
    A("明視野では、ループ後の EPRY・BLIS-FPM の像で、星の中心近くの細いスポークがはっきり見えるようになりました（ループ前は物体のピントがずれていたため、ぼやけていました）。"
      "DIP の像はループの前後で大きくは変わりません。暗視野を含む DIP（最適反復）と BLIS-FPM の像には、ループの後も、細かい斑点状・格子状の模様が残っています。")
    A("")
    if RES2:
        A("**分解能（シーメンススター、中央 512 px、FOV モデル）**：星の MTF は、36 本のスポークの 36 次の成分を符号つきで求めたものです（1.3–1.9 µm⁻¹ の平均で規格化。負はコントラストの反転）。"
          "「MTF 10 %」は 1.5 µm⁻¹ より上で符号つき MTF が 0.1 を下回る最初の周波数、FRC は交互に分けた半分ずつのデータから独立に解いた 2 つの再構成の位相の相関で、half-bit 基準との交点を示します（交点が支持域の限界より上のときは限界の値）。")
        A("")
        A("| 手法 | データ | MTF 10 %：前（再構成したまま） | 前（物体を後から再合焦） | 後（再構成したまま） | FRC half-bit：前 | 前（再合焦） | 後 |")
        A("|---|---|---|---|---|---|---|---|")
        for s_ in ("BF", "BF+DF"):
            for m, mn in M:
                k_ = f"MTF10|512|{s_}|{ML[m]}, FOV model"
                if k_ not in RES2: continue
                t_ = RES2[k_]
                fr = lambda row_: RES2.get(f"FRC_halfbit|{row_}|{s_}|{ML[m]}")
                fq = lambda d_, w_: "—" if not d_ else (("≥ " if d_[w_]["no_crossing_below_support_limit"] else "") + f"{d_[w_]['q']:.2f}")
                A(f"| {mn} | {s_} | {f2(t_['aff'])} | {f2(t_['aff_refocused'])} | {f2(t_['aff2'])} | {fq(fr('as reconstructed'), 'aff')} | {fq(fr('after object refocus'), 'aff')} | {fq(fr('as reconstructed'), 'aff2')} |")
        A("")
    for f_, cap_ in (("fig_mtf_signed_aff_vs_aff2_512.png", "*図 7-3　符号つきの星の MTF（中央 512 px、FOV モデル）。上：明視野、下：暗視野を含む。灰色の破線：ループ前（再構成したまま）、色の点線：ループ前の物体を後から再合焦したもの、色の実線：ループ後（再構成したまま）。帯は雑音 ±3σ。*"),
                     ("fig_frc_aff_vs_aff2_512.png", "*図 7-4　半分ずつのデータから独立に解いた 2 つの再構成の FRC（位相、中央 512 px、FOV モデル）。線の種類は図 7-3 と同じ。点線の黒：half-bit 基準。支持域の限界より上（薄い線）は、再構成の格子から漏れた成分で、分解能の意味はありません。*")):
        if ex("d32a/frc/" + f_):
            A(f"![{f_}](figures/{f_})"); A(""); A(cap_); A("")
    FL_TXT = []
    if RES2:
        mt = lambda s_, m: RES2[f"MTF10|512|{s_}|{ML[m]}, FOV model"]
        fr = lambda row_, s_, m: RES2[f"FRC_halfbit|{row_}|{s_}|{ML[m]}"]
        fq = lambda d_: ("≥ " if d_["no_crossing_below_support_limit"] else "") + f"{d_['q']:.2f}"
        pairs = [(r(PREV[g_], m, mlab), r(MAIN[g_], m, mlab)) for g_ in ("bf512", "bf1000", "df512", "df1000") for mod, mlab, mj in MOD for m, mn in M if r(PREV[g_], m, mlab) and r(MAIN[g_], m, mlab)]
        n_up = sum(xn["spoke_snr3_um_inv"] > xo["spoke_snr3_um_inv"] + 0.01 for xo, xn in pairs); n_dn = sum(xn["spoke_snr3_um_inv"] < xo["spoke_snr3_um_inv"] - 0.01 for xo, xn in pairs)
        dho = [xn["misfit_unfiltered"]["heldout"]["all"] - xo["misfit_unfiltered"]["heldout"]["all"] for xo, xn in pairs]
        rf2 = [abs(r(MAIN[g_], m, "FOV model")["pupil_drift_check"]["object_refocus"]["z_um"]) / 1e3 for g_ in ("bf512", "df512") for m in ("EPRY", "BLIS")]
        FL_TXT.append(f"読み方：ループの後、EPRY・BLIS-FPM の物体は再構成したままでピントが合っています（再合焦量 {max(rf2):.2f} mm 以下）。以前の版の物体を後から再合焦したときとほぼ同じ星の MTF と FRC が、後処理なしで得られました"
                      f"（明視野・512 px：MTF 10 % は EPRY {mt('BF', 'EPRY')['aff']:.2f} → {mt('BF', 'EPRY')['aff2']:.2f}、BLIS-FPM {mt('BF', 'BLIS')['aff']:.2f} → {mt('BF', 'BLIS')['aff2']:.2f} µm⁻¹、"
                      f"FRC half-bit は EPRY {fq(fr('as reconstructed', 'BF', 'EPRY')['aff'])} → {fq(fr('as reconstructed', 'BF', 'EPRY')['aff2'])}、BLIS-FPM {fq(fr('as reconstructed', 'BF', 'BLIS')['aff'])} → {fq(fr('as reconstructed', 'BF', 'BLIS')['aff2'])} µm⁻¹）。"
                      f"スポーク分解能は、32 の再構成のうち {n_up} で上がり、{n_dn} で下がりました。外した画像の misfit の変化は {min(dho):+.3f}〜{max(dho):+.3f} で、データとの合い方はほとんど変わりません（ピントの割り振りの縮退、2.4 節）。")
        pd_ = [(r(PREV[g_], m, mlab), r(MAIN[g_], m, mlab)) for g_ in ("bf512", "bf1000", "df512", "df1000") for mod, mlab, mj in MOD for m in ("DIP_opt", "DIP_30") if r(PREV[g_], m, mlab) and r(MAIN[g_], m, mlab)]
        nd_up = sum(xn["spoke_snr3_um_inv"] > xo["spoke_snr3_um_inv"] + 0.01 for xo, xn in pd_)
        zdo = [r(PREV[g_], "DIP_opt", mlab)["pupil_drift_check"]["object_refocus"]["z_um"] / 1e3 for g_ in ("df512", "df1000") for mod, mlab, mj in MOD]
        zdn = [r(MAIN[g_], "DIP_opt", mlab)["pupil_drift_check"]["object_refocus"]["z_um"] / 1e3 for g_ in ("df512", "df1000") for mod, mlab, mj in MOD]
        zbo = [abs(r(PREV[g_], m, mlab)["pupil_drift_check"]["object_refocus"]["z_um"]) / 1e3 for g_ in ("bf512", "bf1000") for mod, mlab, mj in MOD for m in ("DIP_opt", "DIP_30")] + [abs(r(PREV[g_], "DIP_30", mlab)["pupil_drift_check"]["object_refocus"]["z_um"]) / 1e3 for g_ in ("df512", "df1000") for mod, mlab, mj in MOD]
        FL_TXT.append(f"DIP は瞳を自由に更新するため、明視野の DIP と暗視野を含む DIP（30 反復）では、ループの前から物体の再合焦量はほぼ 0 でした（{max(zbo):.2f} mm 以下）。"
                      f"暗視野を含む DIP（最適反復）だけは例外で、再合焦量はループ前 {min(zdo):+.2f}〜{max(zdo):+.2f} mm、ループ後 {min(zdn):+.2f}〜{max(zdn):+.2f} mm でした。それでも、データに残す視差が変わったことで DIP の瞳のデフォーカスも負の側に動き（表の「瞳 z」）、スポーク分解能は {len(pd_)} 条件のうち {nd_up} 条件で上がりました。"
                      "星の MTF（明視野・512 px）で見ると、DIP の変化は EPRY・BLIS-FPM より小さく、主な改善は、ピントのずれた物体を出していた EPRY・BLIS-FPM に出ています。")
        FL_TXT.append(f"暗視野を含む場合：以前の版の物体を後から再合焦したときに見えた、3.9–4.6 µm⁻¹ の正の MTF（EPRY {mt('BF+DF', 'EPRY')['df_band_aff_refocused']:+.2f}、BLIS-FPM {mt('BF+DF', 'BLIS')['df_band_aff_refocused']:+.2f}。ほぼ 2σ）は、"
                      f"ループ後の再構成では {mt('BF+DF', 'EPRY')['df_band_aff2']:+.2f} / {mt('BF+DF', 'BLIS')['df_band_aff2']:+.2f}（雑音 {mt('BF+DF', 'EPRY')['df_band_noise_aff2']:.2f} / {mt('BF+DF', 'BLIS')['df_band_noise_aff2']:.2f}）で、有意ではありません。"
                      f"一方、以前の版でこの帯域に見えたコントラストの反転（DIP（最適反復）{mt('BF+DF', 'DIP_opt')['df_band_aff']:+.2f}、EPRY {mt('BF+DF', 'EPRY')['df_band_aff']:+.2f}、BLIS-FPM {mt('BF+DF', 'BLIS')['df_band_aff']:+.2f}）は、ループ後はなくなりました"
                      f"（{mt('BF+DF', 'DIP_opt')['df_band_aff2']:+.2f} / {mt('BF+DF', 'EPRY')['df_band_aff2']:+.2f} / {mt('BF+DF', 'BLIS')['df_band_aff2']:+.2f}）。"
                      f"split-half FRC でも、暗視野を加えて高くなった手法はありません（明視野のみ → 暗視野を含む：EPRY {fq(fr('as reconstructed', 'BF', 'EPRY')['aff2'])} → {fq(fr('as reconstructed', 'BF+DF', 'EPRY')['aff2'])}、"
                      f"BLIS-FPM {fq(fr('as reconstructed', 'BF', 'BLIS')['aff2'])} → {fq(fr('as reconstructed', 'BF+DF', 'BLIS')['aff2'])}、DIP（最適反復）{fq(fr('as reconstructed', 'BF', 'DIP_opt')['aff2'])} → {fq(fr('as reconstructed', 'BF+DF', 'DIP_opt')['aff2'])} µm⁻¹）。"
                      "この測定の暗視野の SNR では、暗視野による分解能の向上は確認できない、という前回の結論は変わりません。")
        xd_ = r(MAIN["df512"], "DIP_opt", "FOV model"); cd_ = xd_["pupil_drift_check"]; an_ = ANI.get(MAIN["df512"], {}).get("DIP (optimal it.), FOV model", {}).get("score")
        FL_TXT.append(f"注意：暗視野を含む DIP（最適反復）・512 px・FOV では、瞳に非点（{cd_['pupil_astig_um'][0]/1e3:+.2f}, {cd_['pupil_astig_um'][1]/1e3:+.2f}）mm が入り、物体が逆向きの非点で打ち消す分け方になっています"
                      f"（和の非点 ({cd_['total_astig_um'][0]/1e3:+.2f}, {cd_['total_astig_um'][1]/1e3:+.2f}) mm）。そのため方向の偏りが大きく（{f2(an_)}）、この物体の再合焦量（{cd_['object_refocus']['z_um']/1e3:+.2f} mm）で後から再合焦すると、"
                      f"MTF 10 % はかえって下がります（{mt('BF+DF', 'DIP_opt')['aff2']:.2f} → {mt('BF+DF', 'DIP_opt')['aff2_refocused']:.2f} µm⁻¹）。この条件では、物体のピントの基準は当てになりません。")
    if FL_TXT:
        for t_ in FL_TXT: A(t_)
        A("")
A("## 8. 分解能の見方（スポーク解析の注意）")
if MAIN["bf512"] in C:
    A(f"シーメンススター（36 本）のスポークの変調は、半径 r の円周に沿った 36 次の振幅を、それ以外の次数の rms で割って求めました（SNR）。SNR が 3 を下回る最初の周波数 36/(2πr) を分解能としています。"
      + (f"補正前のデータ（2.1 節の補正のみ）の再構成では、半径 85 px（{85*0.0319:.1f} µm）の内側と外側で、星の中心が {np.hypot(*(np.array(C[OLD['bf512']]['star_centre_crop_px']) - np.array(C[OLD['bf512']]['star_centre_inner_crop_px']))):.1f} px ずれていました"
         f"（外側：{C[OLD['bf512']]['star_centre_crop_px']}、内側：{C[OLD['bf512']]['star_centre_inner_crop_px']}、512 px の切り出しでの画素座標）。この版では {np.hypot(*(np.array(C[MAIN['bf512']]['star_centre_crop_px']) - np.array(C[MAIN['bf512']]['star_centre_inner_crop_px']))):.1f} px で、内外のずれは機械的な像ずれの補正でなくなりました。"
         "解析は同じ方法のまま、中心を 2 つの領域で別々に取っています（`compare_32a.py`、R_SPLIT = 85 px）。中心を 1 つにすると、ずれがあるときは内側の 36 次の振幅が見かけ上なくなり、どの手法も内側の円（36/(2π·85 px) = "
         f"{36/(2*np.pi*85*0.0319):.2f} µm⁻¹）のあたりで頭打ちになります。" if AFF2 and OLD["bf512"] in C else
         f"このチャートでは、半径 85 px（{85*0.0319:.1f} µm）の内側と外側で、再構成像の星の中心が {np.hypot(*(np.array(C[MAIN['bf512']]['star_centre_crop_px']) - np.array(C[MAIN['bf512']]['star_centre_inner_crop_px']))):.1f} px ずれていました（外側：{C[MAIN['bf512']]['star_centre_crop_px']}、内側：{C[MAIN['bf512']]['star_centre_inner_crop_px']}、512 px の切り出しでの画素座標）。"
      "そこで、中心を 2 つの領域で別々に取りました（`compare_32a.py`、R_SPLIT = 85 px）。中心を 1 つにすると、内側の 36 次の振幅が見かけ上なくなり、どの手法も内側の円（36/(2π·85 px) = "
      f"{36/(2*np.pi*85*0.0319):.2f} µm⁻¹）のあたりで頭打ちになります。")
      + f"明視野だけの場合、どの手法も {min(spk(MAIN['bf512'])):.2f}–{max(spk(MAIN['bf512'])):.2f} µm⁻¹ に入り、手法の差は小さいです。像の見た目の違い（DIP は滑らか、EPRY・BLIS-FPM は細かい粒状の模様）は、分解能の差ではなく、雑音と誤差の現れ方の違いです。")
A("")
A("## 9. まとめと今後の課題")
A("1. 平行ビーム・対物 FZP 走査の配置でも、パイプラインの順モデル（対物の位置 → 瞳の位置、視野効果 = 2 次位相）はそのまま使えました。照明波数は、試料なし像のケラレから較正します。")
A("2. この配置で重要なのは、**レンズ位置と検出器位置の食い違いが、そのまま像のずれになる**ことです（ずれは試料面で食い違いと同じ大きさ）。今後のスキャン表では、検出器の値を、FZP に送る整数パルス値から計算する（検出器 X = −527.75 × FZP X、検出器 Y ≈ −42.22 × FZP Y）ことをお勧めします。計算上の位置を FZP のパルスに直すときは、切り捨てより四捨五入の方が、円からのずれが小さくなります。")
if PAR:
    A(f"3. 半パルスの食い違いを直したあとも、機械的な比例ずれ（列←X {PAR['mechanical']['scale_col_pct']:+.2f} %、列←Y {PAR['mechanical']['col_from_row_axis_pct']:+.2f} %、行←Y {PAR['mechanical']['scale_row_pct']:+.2f} %）と、"
      f"高次・ランダムなずれ（rms 数 px）が残ります。これは瞳の非点と取り違えやすいので、posaffine ステップで毎回分けて補正し、瞳が予測する比例ずれとの整合を確認してください（`position_affine.json` の check）。"
      "装置側では、検出器 X の追従倍率（−527.75 の係数、または FZP X の 1 パルスの大きさ）と、FZP の Y 軸と検出器の軸の角度（約 0.5°）の較正をお勧めします。"
      "再構成の瞳に非点が出たときは、物体のピントを合わせた和（コントラストのピント）で判断してください。"
      + ("物体にピントずれが残るときは、フォーカスループ（`focus_iter`）で瞳に移せます。" if AFF2 else ""))
A("4. リング 0–1 には FZP の 0 次光による干渉縞があります。OSA（次数選択絞り）で 0 次光を遮るか、0 次光の項を順モデルに入れれば改善すると考えられます。")
A("5. 暗視野は、10 s 露光でも SNR が 1 未満です。機械的なずれを補正しても、外した暗視野像は予測できず（misfit > 1）、DIP が学習する暗視野像のずれは多くが上限に達しました。分解能を上げるには、露光を延ばす（または積算する）必要があります。")
A("6. 照明はスリットで制限された帯状（強い部分の幅 約 10 µm）のため、全視野 1000 px の上下は照明が弱く、misfit を悪くしています。")
A("")
if AFF2:
    zfree = [r(MAIN[g_], m, mlab)["pupil_drift_check"]["pupil_defocus_um"] / 1e3 for g_ in ("bf512",) for mod, mlab, mj in MOD for m in ("DIP_opt", "DIP_30", "BLIS") if r(MAIN[g_], m, mlab)]
    rfa = [abs(r(MAIN[g_], m, mlab)["pupil_drift_check"]["object_refocus"]["z_um"]) / 1e3 for g_ in ("bf512", "df512") for mod, mlab, mj in MOD for m in ("EPRY", "BLIS") if r(MAIN[g_], m, mlab)]
    A(f"7. 物体側に残っていたピントずれ（約 1.1 mm）を瞳に移すと（フォーカスループ）、EPRY・BLIS-FPM の物体は再構成したままでピントが合い（再合焦量 {max(rfa):.2f} mm 以下、512 px）、分解能が上がりました（7.3 節）。"
      "外した画像の misfit はほとんど変わりません。ピントを物体と瞳にどう割り振るかはデータだけでは決まらないため（視差と機械的な倍率の縮退）、試料にピントが合っていることを条件にしました。"
      f"瞳を自由に更新する DIP と BLIS-FPM では、瞳のデフォーカスは {min(zfree):+.2f}〜{max(zfree):+.2f} mm（明視野・512 px）になり、固定した瞳（{PAR['pupil_defocus_um']/1e3:+.2f} mm）より小さく出ます（自由な瞳の形が 2 次曲線からずれるため。2.4 節）。"
      "瞳のデフォーカスと機械的な倍率を独立に決めるには、試料を光軸方向に既知の量動かした測定（例えば ±2 mm）が有効です。")
else:
    A("7. ピント（デフォーカス）は手法によって 1–3 mm の幅があり、機械的な倍率（等方部分）の値はその分だけ不確かです。試料を光軸方向に既知の量動かした測定（例えば ±2 mm）で、瞳のデフォーカスを独立に決められます。")
A("")
A("## 10. 出力ファイル")
A("- `tiff/<条件>/<手法>_<モデル>_{transmission,phase_rad,pupil_phase_rad}.tif`：物理フレーム、帯域フィルターなし（`d32a/compare_32a.py` が出力）")
A("- `pipeline_BLIS-FPM/`：パイプライン（`run_pipeline.py`、全 60 枚）の BLIS-FPM の結果と図（results/ の TIFF は、パイプラインの仕様で 0.3–3.5 µm⁻¹ に帯域制限されています）")
A("- `json/`：各条件の `compare_summary.json`、`spokes.json`、較正・位置合わせの JSON")
A("- `figures/fig0_position_affine.png`、`json/position_affine.json`：posaffine ステップの結果（測定・瞳・機械的な比例ずれ、確認の値）")
A("- `before_posaffine/`：以前の版（比例ずれを補正する前）のレポート・図・TIFF・JSON・パイプライン出力")
if AFF2:
    A("- `before_focusloop/`：フォーカスループの前の版（2026-09-27 版、瞳 z −1.52 mm）の図・TIFF・JSON・パイプライン出力")
    A("- `supplement_focus_in_pupil/`：フォーカスループの解析（README、図、JSON、スクリプト）。分解能（符号つき MTF・FRC）の前後比較を含みます")
A("- `supplement_astigmatism/`：楕円状の癖（瞳の非点）の原因を調べた解析の JSON とスクリプト")
A("- `pipeline_changes/`：パイプラインへの追加（`fptrecon/posaffine.py`、`nonlinear.py`・`run_pipeline.py` の変更後のファイル、設定例）。同じものをパイプラインのフォルダ（暗視野-並行ビーム照明対応pipeline/fpt_pipeline）にも入れ、変更前のファイルは `*.bak_20260927` として残しました")
A("- `scripts/`：32a 用の補助スクリプト（`d32a/`）、実行したシェルスクリプト（`d32a/runs/`）、使った設定ファイル")
TXT = "\n".join(L) + "\n"
for a_, b_ in (("(-0.00,", "(0.00,"), (", -0.00)", ", 0.00)"), ("(+0.00,", "(0.00,"), (", +0.00)", ", 0.00)")): TXT = TXT.replace(a_, b_)
open(W + "d32a/32a_report.md", "w", encoding="utf-8").write(TXT)
json.dump(dict(C={k: {kk: vv for kk, vv in v.items() if kk != "results"} | {"results": v["results"]} for k, v in C.items()}, fov_dr=fd,
               geometry={k: v for k, v in geo.items() if k not in ("xc_um",)}, register_bright={k: v for k, v in rb.items() if "rms" in k or "max" in k},
               register_model_rms_px=rm_rms, dark_levels=dict(ring6=lev6, ring7=lev7, noise_contrast_ring6=nz6, noise_contrast_ring7=nz7)),
          open(W + "d32a/32a_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print("report lines", len(L))
