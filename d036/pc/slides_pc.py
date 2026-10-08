# -*- coding: utf-8 -*-
# 036 deck: partial-coherence slides.  exec()'d by d036/build_slides_036.py before the 003/018 comparison slide
# (uses new, title, picture, table, textbox, W from there).  Numbers from d036/pc/pc_summary.json.
import json as _json, numpy as _np
_PS = _json.load(open(W + "d036/pc/pc_summary.json")); _T = _PS["tables"]; _B = _PS["blis"]
_MLs = {"DIP (optimal it.)": "DIP 最適", "DIP (30 it.)": "DIP 30 反復", "EPRY": "EPRY", "BLIS-FPM": "BLIS-FPM"}
def _g(tab, m, key="ho1", mod="FOV model"):
    t = _T.get(tab, {}).get("results", {}).get(f"{m}, {mod}"); return None if t is None else t.get(key)
_f3 = lambda v: "—" if v is None else f"{v:.3f}"
_f2 = lambda v: "—" if v is None else f"{v:.2f}"
_sel = _PS["eta_selection"]; _pt = _PS["prediction_test"]; _syn = _PS["synthetic_test"]["results"]; _Sn = _PS["S_nnls"]
_PFIG = {k: W + f"d036/pc/036_fig_pc_{k}.png" for k in ("direct", "compare", "starzoom")}
# P1 ---- model and angular distribution
s = new(); title(s, "部分コヒーレンス（1）：照明の角度広がりを混合状態で表す")
picture(s, _PFIG["direct"], 1.35, 1.0, w=10.6)
_co = _syn["true object | coherent model"]; _pc = _syn["true object | mixed-state model (true S)"]
textbox(s, 0.45, 4.85, 12.4, 2.2, [
    "試料なし画像の直接光は、瞳の縁（約 1.05 k_c）から約 0.4 k_c にわたって 2 桁ゆっくり減衰（a）→ 照明に角度の広がり（CZP の焦点の「すそ」）があると解釈",
    f"照明 = 向きの違う平面波のインコヒーレントな和（1 画像あたり約 20 モード）。S = 中心（幅 {_PS['S_model']['core_sigma_kc']} k_c）＋すそ（割合 η、幅 {_PS['S_model']['halo_ell_kc']} k_c）。試料なし画像だけでは中心の幅は決まらない（b は上限の目安）",
    f"確認：照明 1 本なら従来の明視野像と完全に一致。部分コヒーレントな合成データを混合状態モデルは雑音まで再現（misfit {_pc['misfit_ring1']:.4f}、位相誤差 {100 * _pc['phase_error_rel']:.1f} %）、コヒーレントモデルは {_co['misfit_ring1']:.3f}・{100 * _co['phase_error_rel']:.1f} %",
    f"コヒーレントな物体で S だけ変えた予測（c）：未使用の明視野画像 {_pt['coherent']['ring1 unseen']:.2f} → {_pt['halo eta 0.1']['ring1 unseen']:.2f}、リング 2 {_pt['coherent']['ring2 unseen']:.2f} → {_pt['halo eta 0.1']['ring2 unseen']:.2f}、リング 3 は {_pt['coherent']['ring3 unseen']:.2f} のまま"],
    size=11, space=3)
# P2 ---- eta selection, background, ring 2
s = new(); title(s, "部分コヒーレンス（2）：η の選択と暗視野データのバックグラウンド")
rows = [["BLIS-FPM（28 枚、FOV、4 枚除外）", "学習画像", "未使用 画像 0", "未使用 リング 3"]]
for k, lab in (("1mode", "照明 1 本"), ("0.1F", "η = 0.1（採用）"), ("0.3F", "η = 0.3")):
    h = _sel[k]["heldout_per_image"]
    rows.append([lab, f"{_sel[k]['misfit']:.4f}", f"{h[0]:.3f}", " / ".join(f"{v:.3f}" for v in h[1:])])
table(s, rows, 0.6, 1.25, 7.6, col_w=[2.9, 1.3, 1.4, 2.0], size=11, row_h=0.38)
_ng = _PS["prediction_test_nogain"]["coherent"]
_r2p, _r2c = _B.get("r123/s-1_ho_pc0.1A.json"), _B.get("r123/s-1_ho_pc1mode.json")
_r2txt = "リング 2 を含む 44 枚の BLIS-FPM は計算中"
if _r2p and _r2c:
    _hi = _r2p["heldout_images"]; _i2 = [j for j, i in enumerate(_hi) if 9 <= i <= 24]; _i3 = [j for j, i in enumerate(_hi) if i >= 25]
    _fm = lambda d, js: " / ".join(_f3(d["heldout_per_image"][j]) for j in js)
    _r2txt = (f"44 枚（BLIS-FPM）：学習したリング 2 は {_r2c['per_ring']['2']:.3f} → {_r2p['per_ring']['2']:.3f} と大きく改善。"
              f"未使用のリング 2 は {_fm(_r2c, _i2)} → {_fm(_r2p, _i2)}、リング 3 は {_fm(_r2c, _i3)} → {_fm(_r2p, _i3)}（モードの表し方による不確かさ大）")
_r1p, _r1c = _B.get("r1/s-1_ho_pc0.1F.json"), _B.get("r1/s-1_ho_pc1mode.json")
_r1txt = ""
if _r1p and _r1c:
    _r1txt = (f"明視野だけの 9 枚（BLIS-FPM）：学習画像 {_r1c['misfit']:.3f} → {_r1p['misfit']:.3f} だが、未使用の画像は "
              f"{' / '.join(_f3(v) for v in _r1c['heldout_per_image'])} → {' / '.join(_f3(v) for v in _r1p['heldout_per_image'])} とわずかに悪化（学習 7 枚では当てはめすぎ）")
textbox(s, 0.6, 3.05, 7.6, 3.9, [
    "η = 0.1 と 0.3 は未使用画像では区別できない → リング 2 の予測がわずかに良い 0.1 を採用",
    f"暗視野データには、モデルで説明できない一様なバックグラウンド（リング 2 でモデル強度の約 10 倍、リング 3 で約 2.6 倍）→ 暗視野の画像ごとに倍率（DIP・EPRY はオフセットも）を当てはめる。平均で割るだけだとリング 2 の予測 misfit は {_ng['ring2 unseen']:.0f}",
    "比較用の「照明 1 本」も同じプログラム・同じ扱いで計算",
    _r2txt] + ([_r1txt] if _r1txt else []), size=11, space=4)
textbox(s, 8.5, 1.25, 4.4, 5.7, [("h", "副次的に分かったこと"),
    "FOV の二次位相を周期的な格子にのせると、格子の端で位相が飛び、見かけの直接光ができる（リング 3 では、窓の端にある暗視野強度の平均の約 1/3）。従来の結果は窓関数でほぼ抑えられている",
    "BLIS-FPM では、混合状態の直接光を毎回計算し直す必要がある（瞳の位相で直接光の形が大きく変わる）",
    "EPRY は照明 1 本でも部分コヒーレントでも、未使用画像の misfit が最小になるのは 1〜5 回目の掃引"], size=11, space=5)
# P3 ---- 4-method comparison
s = new(); title(s, "部分コヒーレンス（3）：4 手法の比較（28 枚、FOV モデル）")
picture(s, _PFIG["compare"], 0.35, 1.05, w=8.3)
_imp512 = [100 * (1 - _g("512_pc", m) / _g("512_pc1m", m)) for m in _MLs if _g("512_pc", m) and _g("512_pc1m", m)]
_imp1024 = [100 * (1 - _g("1024_pc", m) / _g("1024_pc1m", m)) for m in _MLs if _g("1024_pc", m) and _g("1024_pc1m", m)]
textbox(s, 8.85, 1.15, 4.1, 5.8, [
    f"未使用の明視野画像（画像 0）：照明 1 本に比べ 4 手法とも改善（512 画素 {min(_imp512):.0f}–{max(_imp512):.0f} %、1024 画素 {min(_imp1024):.0f}–{max(_imp1024):.0f} %）",
    f"DIP 最適・512 の元の計算（4 章）は {_f3(_g('512_dfgain', 'DIP (optimal it.)'))} で、部分コヒーレントの {_f3(_g('512_pc', 'DIP (optimal it.)'))} と同程度。1024 では {_f3(_g('1024_dfgain', 'DIP (optimal it.)'))} → {_f3(_g('1024_pc', 'DIP (optimal it.)'))}",
    f"未使用のリング 3：4 手法とも {min(_g(t, m, 'ho3') for t in ('512_pc', '1024_pc') for m in _MLs):.2f}–{max(_g(t, m, 'ho3') for t in ('512_pc', '1024_pc') for m in _MLs):.2f} で予測できないまま",
    "DIP 1024 は FOV モデルのみ（計算時間のため）"], size=11.5, space=6)
# P4 ---- star centre and conclusions
s = new(); title(s, "部分コヒーレンス（4）：星の中心の位相と結論")
picture(s, _PFIG["starzoom"], 0.35, 1.15, w=8.5)
textbox(s, 9.1, 1.2, 3.9, 5.7, [
    "スポーク SNR > 3（1024、FOV）：" + "、".join(f"{_MLs[m]} {_f2(_g('1024_pc1m', m, 'spoke_snr3_um_inv'))} → {_f2(_g('1024_pc', m, 'spoke_snr3_um_inv'))}" for m in _MLs) + " µm⁻¹",
    f"分解能は EPRY 以外変わらない（EPRY の差は、選ばれた掃引の回数 {_T['1024_pc1m']['results']['EPRY, FOV model']['iteration']} と {_T['1024_pc']['results']['EPRY, FOV model']['iteration']} の違い）",
    (("h", "結論")),
    "照明の角度広がりは実在し、暗視野を含む 28 枚では明視野の当てはめと予測をわずかに良くする（明視野だけの 9 枚では予測は良くならない）",
    "リング 3 の強度を予測できない原因は、角度広がりではない（別の系統誤差か雑音）",
    "リング 2 は照明モードの表し方に敏感で、まだ確定的なことは言えない"], size=11.5, space=5)
