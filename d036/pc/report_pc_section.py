# -*- coding: utf-8 -*-
# 036 report, section 5 (partial coherence).  exec()'d by d036/make_report_036.py after section 4; uses A() and W from there.
# Every number comes from d036/pc/pc_summary.json (summarize_pc.py).
import json as _json, numpy as _np
PS = _json.load(open(W + "d036/pc/pc_summary.json"))
_T = PS["tables"]; _B = PS["blis"]
_ML = {"DIP (optimal it.)": "DIP（最適反復）", "DIP (30 it.)": "DIP（30 反復）", "EPRY": "EPRY", "BLIS-FPM": "BLIS-FPM"}
_M = list(_ML)
def _g(tab, m, mod="FOV model", key="ho1"):
    t = _T.get(tab, {}).get("results", {}).get(f"{m}, {mod}")
    return None if t is None else t.get(key)
f3_ = lambda v: "—" if v is None else f"{v:.3f}"
f2_ = lambda v: "—" if v is None else f"{v:.2f}"
def _pct(a, b):  # relative change a -> b in %
    return None if (a is None or b is None) else 100 * (b - a) / a
sel = PS["eta_selection"]; pt = PS["prediction_test"]; syn = PS["synthetic_test"]["results"]; Sn = PS["S_nnls"]
A("## 5. 部分コヒーレンス（照明の角度広がり）を考慮した再構成\n")
A("照明を一つの平面波ではなく、少しずつ向きの違う平面波の**インコヒーレントな和**（混合状態）として扱う順モデルを、3 つのパイプラインに入れました。"
  "各画像の強度は、照明モード j（向き k_n + Δk_j、重み w_j）ごとの像の強度の重み付き和 I_n = Σ_j w_j |F⁻¹{P(k) Ô(k − k_n − Δk_j)}|² です。"
  "物体・瞳・FOV の曲率はすべてのモードで共通です。明視野の画像は同じモデルで計算した直接光で割り、暗視野の画像は（データから試料なし画像を引いてあるので）モデルでも直接光を引きます。"
  "照明を 1 本にすれば従来のコヒーレントな計算と同じになります（明視野の像は DIP・EPRY・BLIS-FPM とも完全に一致）。\n")
A("### 5.1 照明の角度分布の見積もり\n")
A(f"- **試料なし画像**：FOV の曲率のため、視野内の位置ごとに照明の局所的な向きが少しずつ違います。これを使うと、リング 1・2 の 25 枚の試料なし画像を一本の曲線「局所的な照明の向き → 直接光の強さ」にまとめられます（図 6a）。"
  f"直接光は瞳の縁（約 1.05 k_c）で急に落ちたあと、**約 0.4 k_c にわたって 2 桁ほどゆっくり減衰**します（明視野の強さに対して、1.2 k_c で約 0.1、1.45 k_c で約 0.01、1.8 k_c で約 0.003）。"
  "FZP の開口の縁は鋭いはずなので、この裾は照明の角度の広がり（CZP の焦点の「すそ」）と解釈しました。")
A(f"- **角度分布 S の形**：試料なし画像だけでは、S の中心部の幅は決まりません（c = 1 では縁の近く 1.0–1.1 k_c を通る画素がなく、試料なし画像の強さや照明角の誤差とも区別できないため）。"
  f"試料なし画像から逆算した S（上限の目安）は、照明の {100 * Sn['encircled']['0.05']:.0f} % が 0.05 k_c 以内、{100 * Sn['encircled']['0.1']:.0f} % が 0.1 k_c 以内、{100 * Sn['encircled']['0.5']:.0f} % が 0.5 k_c 以内です。"
  f"再構成では、S = 中心成分（幅 {PS['S_model']['core_sigma_kc']} k_c のガウス分布）＋ すそ（割合 η、2 次元の Cauchy 型、幅 {PS['S_model']['halo_ell_kc']} k_c、指数 {PS['S_model']['halo_beta']:g}）とし、η は試料像で決めました（5.3）。")
A("- **時間的コヒーレンス**：単色器の帯域を Δλ/λ ≈ 1.3 × 10⁻⁴（Si 111 相当、仮定）とすると、FZP の焦点距離の変化は約 0.1 mm で、焦点深度（λ/NA² ≈ 3.9 mm）よりずっと小さいので無視しました。")
A("- **照明モードの数**：S を細かく（147 モード）表したものを基準に、画像ごとにモードをまとめる方法を試しました。中心付近（0.08 k_c 以内）のモードはそのまま残し、それより外側は「瞳の内側に入るか外側か」と方位で束ねます。"
  "こうすると 1 画像あたり平均約 20 モードで、リング 1・3 の像は基準との差が約 1 % 以内です。リング 2 の像は瞳の縁に近いモードに敏感で、33 モードでも基準との差が約 17 % 残りました。\n")
A("### 5.2 モデルの確認\n")
_st = [k for k in syn if k.startswith("true object |")]
_co = syn["true object | coherent model"]; _pcs = syn["true object | mixed-state model (true S)"]
A(f"- **合成データ**：部分コヒーレント（η = 0.3）で作った 28 枚の合成データ（視野 8 µm、雑音 1 %）を、正しい物体から出発して解き直しました。"
  f"混合状態モデルはデータを雑音の水準まで再現し（リング 1 の misfit {_pcs['misfit_ring1']:.4f}）、位相は正解から {100 * _pcs['phase_error_rel']:.1f} % しかずれません。"
  f"コヒーレントモデルは misfit が {_co['misfit_ring1']:.3f} より下がらず、位相が正解から {100 * _co['phase_error_rel']:.1f} % ずれました。")
A(f"- **コヒーレントな物体での予測**：コヒーレントな BLIS-FPM の解（28 枚、4 枚を除いて解いたもの）の物体と瞳をそのまま使い、S だけを変えて全 44 枚を予測しました（暗視野の画像は倍率を 1 つ当てはめる。図 6c）。"
  f"使っていない明視野の画像 0 の misfit は、コヒーレント {pt['coherent']['ring1 unseen']:.3f} に対して、部分コヒーレントでは {min(v['ring1 unseen'] for k, v in pt.items() if k != 'coherent'):.3f}–{max(v['ring1 unseen'] for k, v in pt.items() if k != 'coherent'):.3f} に下がりました。"
  f"一度も使っていないリング 2 の 16 枚は {pt['coherent']['ring2 unseen']:.3f} → {pt['halo eta 0.1']['ring2 unseen']:.3f}（η = 0.1）、リング 3 の 3 枚は {pt['coherent']['ring3 unseen']:.3f} → {pt['halo eta 0.1']['ring3 unseen']:.3f} です。")
_ng = PS["prediction_test_nogain"]["coherent"]
A(f"- **暗視野データのバックグラウンド**：暗視野の画像をそれぞれの平均で規格化するだけだと、リング 2 の予測の misfit は {_ng['ring2 unseen']:.1f} と 1 を大きく超えました。"
  "データの平均がモデルの暗視野強度よりずっと大きい（リング 2 で約 10 倍、リング 3 で約 2.6 倍）ためで、モデルで説明できない一様なバックグラウンドが暗視野データに乗っています。"
  "そこで以下では、暗視野の画像ごとにコントラストの倍率を 1 つ最小二乗で当てはめます（DIP は実空間の損失なので倍率とオフセット、EPRY は最初の更新時に倍率とオフセットを推定して固定）。"
  "この扱いは比較用のコヒーレント計算（照明 1 本）にも同じように入れています。")
A("- **コヒーレントな計算の副作用**：FOV の二次位相を周期的な計算格子にのせると、格子の端で位相が不連続になり、そこから瞳に入る見かけの直接光ができます。"
  "リング 3 では、窓関数の端（重み 0 の領域）にある暗視野強度の平均の約 1/3 に達していました。従来の結果は窓関数でほぼ抑えられていますが、画像の平均で規格化するときに入っていました。"
  "混合状態の計算では、平均を窓関数の重み付きで取ることでこの影響を除いています。\n")
A("### 5.3 すその割合 η の選択\n")
A("BLIS-FPM（28 枚、FOV、画像 0・25・34・43 を除いて解き直し）で、照明 1 本・η = 0.1・η = 0.3 を比べました。\n")
A("| モデル | 学習画像の misfit | 未使用：画像 0（リング 1） | 未使用：リング 3（25・34・43） |\n|---|---|---|---|")
for k, lab in (("1mode", "照明 1 本（コヒーレント）"), ("0.1F", "部分コヒーレント η = 0.1"), ("0.3F", "部分コヒーレント η = 0.3")):
    h = sel[k]["heldout_per_image"]
    A(f"| {lab} | {sel[k]['misfit']:.4f} | {h[0]:.3f} | {' / '.join(f'{v:.3f}' for v in h[1:])} |")
A("")
_m = {k: float(_np.mean(sel[k]["heldout_per_image"])) for k in sel}
A(f"部分コヒーレンスで学習画像の misfit は {-_pct(sel['1mode']['misfit'], sel['0.1F']['misfit']):.0f}–{-_pct(sel['1mode']['misfit'], sel['0.3F']['misfit']):.0f} % 下がり、未使用の明視野の画像も良くなりました。"
  f"η = 0.1 と 0.3 は未使用画像では区別できない（4 枚の平均 {_m['0.1F']:.3f} と {_m['0.3F']:.3f}）ので、リング 2 の予測がわずかに良かった **η = 0.1** を以下で使いました。\n")
A("### 5.4 4 手法の比較（28 枚、FOV モデル）\n")
A("「照明 1 本」は、同じプログラム・同じ暗視野の倍率の扱いで照明を 1 本にした比較用の計算です。「元の計算」は 4 章のコヒーレントな結果を同じ指標で採点し直したものです。"
  "BLIS-FPM の値は 4 枚を除いて解き直した解、それ以外の手法は未使用画像をもともと学習に使っていません。1.0 は「何も予測しない」のと同じです。\n")
for G, lab in ((512, "中央 512 画素"), (1024, "全視野 1024 画素")):
    if f"{G}_pc" not in _T: continue
    A(f"#### {lab}\n")
    A("| 手法 | 未使用 画像 0：元の計算 / 照明 1 本 → 部分コヒーレント | 学習 リング 1：照明 1 本 → 部分コヒーレント | 未使用 リング 3：照明 1 本 → 部分コヒーレント |" + (" スポーク SNR > 3（µm⁻¹）：照明 1 本 → 部分コヒーレント |" if G == 1024 else "") +
      "\n|---|---|---|---|" + ("---|" if G == 1024 else ""))
    for m in _M:
        row = (f"| {_ML[m]} | {f3_(_g(f'{G}_dfgain', m))} / {f3_(_g(f'{G}_pc1m', m))} → **{f3_(_g(f'{G}_pc', m))}** | {f3_(_g(f'{G}_pc1m', m, key='fit1'))} → {f3_(_g(f'{G}_pc', m, key='fit1'))} | "
               f"{f3_(_g(f'{G}_pc1m', m, key='ho3'))} → {f3_(_g(f'{G}_pc', m, key='ho3'))} |")
        if G == 1024:
            row += f" {f2_(_g(f'{G}_pc1m', m, key='spoke_snr3_um_inv'))} → {f2_(_g(f'{G}_pc', m, key='spoke_snr3_um_inv'))} |"
        A(row)
    A("")
_impG = {G: [-_pct(_g(f"{G}_pc1m", m), _g(f"{G}_pc", m)) for m in _M if _g(f"{G}_pc1m", m) is not None and _g(f"{G}_pc", m) is not None] for G in (512, 1024)}
_h3 = [_g(t, m, key='ho3') for t in ('512_pc', '1024_pc') for m in _M if _g(t, m, key='ho3') is not None]
A(f"- **明視野（リング 1）**：照明 1 本と比べると、部分コヒーレンスで未使用の画像 0 の misfit は 4 手法とも下がりました（中央 512 画素で {min(_impG[512]):.0f}–{max(_impG[512]):.0f} %、全視野 1024 画素で {min(_impG[1024]):.0f}–{max(_impG[1024]):.0f} %）。"
  f"ただし中央 512 画素の DIP（最適反復）では、元の計算が {f3_(_g('512_dfgain', 'DIP (optimal it.)'))} で、部分コヒーレントの {f3_(_g('512_pc', 'DIP (optimal it.)'))} と同程度です。"
  "この照明 1 本の計算が元の計算より悪いのは、暗視野の扱い（オフセット、直接光の差し引き、反復数 300）を変えたためで、部分コヒーレンスの効果ではありません。"
  f"全視野 1024 画素の DIP（最適反復）では、元の計算 {f3_(_g('1024_dfgain', 'DIP (optimal it.)'))} に対して部分コヒーレントで {f3_(_g('1024_pc', 'DIP (optimal it.)'))} に下がりました。")
A(f"- **暗視野（リング 3）**：未使用のリング 3 の画像は、部分コヒーレンスを入れても 4 手法・両方の視野とも 1 に近いまま（{min(_h3):.2f}–{max(_h3):.2f}）で、予測できていません。"
  "リング 3 の強度を予測できない主な原因は、ここで入れた照明の角度広がりではないと考えられます。")
# ring 2 (44 images) and bright only (9 images), BLIS-FPM full field
def _bl(key): return _B.get(key)
r2p, r2c = _bl("r123/s-1_ho_pc0.1A.json"), _bl("r123/s-1_ho_pc1mode.json")
r2pf, r2cf = _bl("r123/s-1_pc0.1A.json"), _bl("r123/s-1_pc1mode.json")
if r2p and r2c:
    hi = r2p["heldout_images"]; grp = lambda lo, up: [j for j, i in enumerate(hi) if lo <= i <= up]
    fmt = lambda d, js: " / ".join(f3_(d["heldout_per_image"][j]) for j in js)
    i1_, i2_, i3_ = grp(0, 8), grp(9, 24), grp(25, 43)
    A(f"- **リング 2 を含めた 44 枚（BLIS-FPM、FOV、モードの分け方 A、平均 {r2p['n_modes']:.0f} モード）**："
      + (f"全 44 枚で解いた misfit は照明 1 本 {r2cf['misfit']:.4f} → 部分コヒーレント {r2pf['misfit']:.4f}、リング 2 の画像の平均は {r2cf['per_ring']['2']:.3f} → {r2pf['per_ring']['2']:.3f} です。" if (r2pf and r2cf) else "")
      + f"画像 {', '.join(map(str, hi))} を除いて解き直すと、学習に使ったリング 2 は {r2c['per_ring']['2']:.3f} → {r2p['per_ring']['2']:.3f}、"
      f"使っていない画像は、リング 1（画像 {', '.join(str(hi[j]) for j in i1_)}）{fmt(r2c, i1_)} → {fmt(r2p, i1_)}、"
      f"リング 2（画像 {', '.join(str(hi[j]) for j in i2_)}）{fmt(r2c, i2_)} → {fmt(r2p, i2_)}、"
      f"リング 3（画像 {', '.join(str(hi[j]) for j in i3_)}）{fmt(r2c, i3_)} → {fmt(r2p, i3_)} でした。"
      "部分コヒーレンスはリング 2 の当てはめを大きく良くしますが（直接光のすそが境界の画像に入るため）、使っていないリング 2 の予測の改善は小さく、リング 3 は良くなりません。"
      "リング 2 は照明モードのまとめ方で像が大きく変わる（5.1）ので、この数値の不確かさは大きめです。")
r1p, r1c = _bl("r1/s-1_ho_pc0.1F.json"), _bl("r1/s-1_ho_pc1mode.json")
r1pf, r1cf = _bl("r1/s-1_pc0.1F.json"), _bl("r1/s-1_pc1mode.json")
r1pp, r1cp = _bl("r1/s+0_ho_pc0.1F.json"), _bl("r1/s+0_ho_pc1mode.json")
if r1p and r1c:
    A(f"- **明視野のみ 9 枚（BLIS-FPM）**："
      + (f"9 枚すべてで解いた misfit は FOV で {r1cf['misfit']:.4f} → {r1pf['misfit']:.4f}。" if (r1pf and r1cf) else "")
      + f"画像 {' と '.join(map(str, r1p['heldout_images']))} を除いて解き直すと、学習画像は {r1c['misfit']:.4f} → {r1p['misfit']:.4f} と下がる一方、"
      f"使っていない画像は {' / '.join(f3_(v) for v in r1c['heldout_per_image'])} → {' / '.join(f3_(v) for v in r1p['heldout_per_image'])} とわずかに悪くなりました"
      + (f"（平面波モデルでも {' / '.join(f3_(v) for v in r1cp['heldout_per_image'])} → {' / '.join(f3_(v) for v in r1pp['heldout_per_image'])}）" if (r1pp and r1cp) else "")
      + "。学習画像が 7 枚しかないと、部分コヒーレンスで増えた表現の自由度が学習画像への当てはめすぎに使われると考えられます。"
      "未使用の明視野画像の予測が良くなったのは、暗視野の画像で物体と瞳が強く拘束される 28 枚の場合です。")
A("")
A("### 5.5 まとめ\n")
A("- 照明の角度広がりは実在し（試料なし画像の直接光のすそ）、混合状態のモデルで扱うと、暗視野を含む 28 枚では明視野の当てはめと予測がわずかに良くなります。明視野だけの 9 枚（BLIS-FPM）では、学習画像の当てはめは良くなるものの予測は良くなりません。")
A("- リング 3 の強度を予測できないのは、角度広がりのためではありません。別の系統誤差（照明角、暗視野のバックグラウンド、検出器）か雑音が原因と考えられます。")
A("- 暗視野の画像には一様なバックグラウンドがあり、強度の絶対値をそのまま当てはめることはできませんでした。")
A("- リング 2 は照明モードの表し方に敏感で、ここでの結果はまだ確定的ではありません。\n")
A("![部分コヒーレンス：直接光と予測](036_fig_pc_direct.png)\n\n**図 6** (a) 25 枚の試料なし画像から求めた、局所的な照明の向きと直接光の強さの関係（FOV の曲率で位置を向きに換算、c = 1.5 の当てはめ）。(b) 角度分布 S の候補（すその割合 η）と、試料なし画像から逆算した S（上限の目安）。(c) コヒーレントな物体で S だけを変えたときの、使っていない画像の misfit。\n")
A("![部分コヒーレンス：4 手法の比較](036_fig_pc_compare.png)\n\n**図 7** 照明 1 本（灰）と部分コヒーレント（色）の比較（28 枚、FOV モデル）。未使用の明視野の画像 0、学習に使ったリング 1、未使用のリング 3 の misfit。◇ は 4 章の元の計算。\n")
A("![部分コヒーレンス：星の中心](036_fig_pc_starzoom.png)\n\n**図 8** 星の中心の位相（全視野 1024 画素、帯域制限なし、FOV モデル）。上段：照明 1 本、下段：部分コヒーレント。円は 36 本のスポークの変調が SNR 3 を下回る半径。\n")
