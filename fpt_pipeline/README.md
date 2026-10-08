# fptrecon — BLIS-FPM による X 線フーリエタイコグラフィ再構成パイプライン

再構成法の名称は **BLIS-FPM**（Band-Limited Intensity-Spectrum FPM、帯域制限強度スペクトル FPM）です。パッケージ名 `fptrecon` とコマンドは、既存のスクリプトや設定との互換性のため変えていません。

FZP 対物の結像型 X 線顕微鏡で撮影した多角度照明データ（CZP の走査で照明角を変えたもの）から、**透過像・位相像・瞳関数**を再構成する。視野の影響（非テレセントリック FZP による視野位置依存の照明角）は、モデルに入れる場合と入れない場合の両方を計算できる。

## 処理の流れ
| step | 内容 | 主な出力（`<output_dir>/work/`） |
|---|---|---|
| `preprocess` | ダーク減算 → 繰り返し撮影の平均 → 試料/直接光でフラット補正 → 平均 1 に正規化 → ホット画素除去 → 低周波平坦化（σ = 30 px）。繰り返し撮影の差分から雑音スペクトルも算出する | `preprocessed.npy`, `preprocess_meta.npz` |
| `calibrate` | 中心 512 px の強度スペクトルに弱物体伝達関数（WOTF）モデルをフィットする。求めるのは、ステージ→画像の回転と鏡映、角度換算係数、瞳中心オフセット k0、デフォーカス、非点収差。探索順は格子探索 → 差分進化 + 多点 Nelder–Mead → アフィン精密化 | `calibration.json` |
| `posaffine`（任意） | 像のずれを隣り合う画像の相互相関から測り、照明位置に比例する部分（2×2 行列：倍率・回転・せん断）・高次・ランダムに分解する。瞳が予測する視差だけを残し、残りを機械的な誤差として画像から補正する。瞳のデフォーカス・非点は、物体のピントを合わせた和（コントラストのピント）で決める。位置を固定して解き直し、瞳が予測する比例ずれと残したずれが合うか確認する | `preprocessed.npy`（補正後）、`preprocessed_before_posaffine.npy`、`calibration.json`、`results/position_affine.json`、`figures/fig0_position_affine.png` |
| `linear` | 重なりタイル（256 px、50 %）上で正則化付き WOTF 線形解を求める（非線形計算の初期値）。吸収と位相の相関の符号から双対解を判定する | `linear_c*.npz` |
| `nonlinear` | **BLIS-FPM の本体。** 帯域制限（0.3–3.5 µm⁻¹）した強度スペクトル最小二乗を L-BFGS で解き、物体 (a, φ) と瞳位相 W(k) を同時推定する。モデル強度は直接光モデルで割り、データと同じくフラット補正された量として扱う | `nonlinear_c*.npz` |
| `export` | 物理フレームへの変換（双対解の選択）、帯域制限、カメラ格子への再標本化、TIFF・Zernike 係数・タイル別ミスフィットの出力 | `results/` |
| `figures` | 較正・再構成・瞳・視野効果の図 | `figures/` |

**視野効果**は O_eff(x) = O(x)·exp(iπκ|x|²)、κ = c/(λp) で表す。`fov_factors` に c を並べると、それぞれの c で再構成する。c = 0 は平面波モデル（視野の影響なし）、c = 1 は公称の p（FZP 主光線による効果）に対応する。

## 環境
```bash
conda env create -f environment.yml      # numpy scipy tifffile matplotlib pytorch (CPU)
conda activate fptrecon
```

## 使い方
```bash
# 全ステップ（c = 0 と 1）
python run_pipeline.py --config config_003.json
# 動作確認（較正を粗くし、非線形は 3+5 反復のみ）
python run_pipeline.py --config config_003.json --quick
# 前処理・較正は再利用して、非線形以降だけやり直す
python run_pipeline.py --config config_003.json --steps nonlinear,export,figures
# 視野効果の強さを走査する（linear から実行し直す）
python run_pipeline.py --config config_003.json --steps linear,nonlinear,export,figures --fov-factors 0,0.5,1,1.5
# 反復回数を変える
python run_pipeline.py --config config_003.json --steps nonlinear,export,figures --n-pupil 20 --n-joint 150
```
所要時間の目安（8 コア CPU、1500×1500×44 枚）：preprocess 約 1 分、calibrate 約 15 分、linear は c 1 つあたり約 20 秒、nonlinear は c 1 つあたり 170 反復で約 25–35 分。

## 設定ファイル（JSON）の主な項目
| key | 意味 | 003 の値 |
|---|---|---|
| `data_dir`, `sample_subdir`, `direct_subdir`, `positions_csv` | データの場所（CSV の列は idx, x[pulse], y[pulse], …） | Google Drive 上のフォルダ |
| `n_repeat`, `repeat_order` | 繰り返し撮影の枚数と並び順（`interleaved`: a001, a002 が位置 1） | 2, interleaved |
| `energy_keV`, `pixel_um`, `fzp_outer_zone_um` | λ、試料面換算画素、NA = λ/(2Δr) | 30, 0.0319, 0.100 |
| `sample_objective_distance_m` | p（視野効果の κ_nom = 1/(λp)） | 0.75 |
| `czp_focal_m`, `stage_step_um` | 公称の角度換算係数（較正の初期値にだけ使う） | 2.4, 0.5 |
| `dark_offset` | ダーク値（counts） | 100（推定値） |
| `calib.*` | 較正の探索範囲 | 既定値 |
| `nonlinear.max_k_over_kc` | この値を超えて NA 縁に近い照明は除外する | 0.97 |
| `nonlinear.M`, `nonlinear.No` | 低分解能パッチと物体格子の大きさ（null なら自動） | 672, 1200 |
| `nonlinear.n_pupil`, `n_joint` | 瞳のみの反復数と、物体+瞳の同時反復数 | 20, 150 |
| `fov_factors` | 視野効果の係数 c のリスト | [0, 1] |

## 出力（`<output_dir>/results/`）
- `planewave_*`（c = 0）、`FOV_cp1_00_*`（c = 1）など
  - `*_transmission.tif`, `*_phase_rad.tif`：1500×1500 float32（31.9 nm/px）。帯域 0.3–3.5 µm⁻¹、物理フレーム
  - `*_pupil_phase_rad.tif`, `*_pupil_amplitude.tif`：瞳（0.0209 µm⁻¹/px、|k| ≤ 1.08 k_c の範囲を切り出し）。振幅は NA の円盤に固定しており、推定していない
  - `*_summary.json`：ミスフィット、Zernike 係数、デフォーカス換算値、6×6 タイル別ミスフィット
- `calibration_physical.json`：物理フレームでの照明 k、アフィン行列、k0、デフォーカス、各 c のミスフィット

## 規約
- 照明は exp(+i2πk·x)、k = (row, col)。物体は O = exp(−a + iφ) で、物質では φ < 0。瞳は P = disc·exp(iW) で、デフォーカスは W = πλz|k|²。
- 強度データだけでは、双対解 (O, P(k), k) ↔ (O*, P*(−k), −k) と視野効果の係数の符号 κ ↔ −κ を区別できない。本パイプラインは線形解で corr(a, φ) < 0 となる側を物理解として選び、κ の符号もそれに合わせる。

## 注意点
1. 較正の解は多峰的である。`fig1_calibration.png` 上段で、瞳円（シアンの破線）が各スペクトルの形（NA 縁の照明で現れる砂時計形）と一致しているか必ず確認すること。ログには多点探索の各候補も出力される。
2. 出力は、データの SNR で制限された帯域だけを含む。0.3 µm⁻¹ 未満（位相の絶対値・膜厚換算）と 3.5 µm⁻¹ 超は復元していない。
3. ダーク値・p・CZP 焦点距離が実測できる場合は、設定ファイルに入れること。
4. 部分コヒーレンス、視野内のデフォーカス変化、瞳振幅は現モデルに含まれていない。

## 018（ITEX 形式・1 回撮影）への対応（2026-09-26 追加）
- 入力：`sample_subdir` / `direct_subdir` に TIFF がなければ、浜松ホトニクス ITEX 形式（`*.img`、16 bit）を読みます（`fptrecon/preprocess.py` の `read_itex`）。
- `n_repeat = 1`（各位置 1 回撮影）では、雑音スペクトルを |q| > 2.1 k_c の平坦部から推定します。003 のデータで確かめると、3–4 µm⁻¹ の雑音を 10–20 % 低く見積もります。
- `twin`：位相の符号（双対解）の手動指定。`null`（既定）は従来どおり線形解の corr(a, φ) で自動判定、`+1` / `-1` で固定します。判定の出どころは `calibration.json` の `twin_source` に残ります。018 では `-1` に固定しました（理由は `config_018.json` の `_comment_twin` と 018 レポート §2）。
- `fptrecon/postprocess.py`：タイル幅が奇数のときにタイル別ミスフィットの計算が落ちる不具合を直しました（003 の結果は変わりません）。
- 018 の実行例：`python run_pipeline.py --config config_018.json`（1024×1024×44 枚で、前処理＋較正約 18 分、非線形は c 1 つあたり 170 反復で約 6–12 分）。

## 036 暗視野モード（2026-09-26 追加）
- `fptrecon/nonlinear.py` の `FPMBand` に、省略可能な引数を 3 つ追加しました。すべて `None`（既定）なら従来と同じ動作で、003・018 の結果は変わりません。
  - `dark`：(n,) bool。直接光が瞳の外に出る画像（暗視野）では、直接光による規格化をしません。
  - `wmask`：(n, M, M)。画素ごとの有効重み（0–1）。窓関数に掛け、コントラストは重み付き平均のまわりで取ります。
  - `img_weights`：(n,)。損失での画像ごとの重み λ_n（例：雑音パワーの逆数）。
- `run_pipeline.py` 自体には暗視野モードはありません。036 の暗視野を含む BLIS-FPM は `d036/blis_df.py` から `FPMBand` を呼んで解いています（使い方は一式の `README_036_暗視野パイプライン.md`）。
- 036 の明視野（リング 1 の 9 枚）は `config_036bf.json` で、従来どおり `run_pipeline.py` で解きます。

## 036 部分コヒーレンス（2026-09-26 追加）

- `fptrecon/coherence.py`：`pc_modes(pc, kn, kc, r_p)` が、設定 `pc`（例 `{"eta": 0.1, "modes": "F", "sigma": 0.02, "ell": 0.1, "beta": 2.0}`）から画像ごとの照明モード `modes`（n, J, 2、k_c 単位の向き）と重み `mode_w`（n, J）を返します。`"eta": "1mode"` なら照明 1 本。
- `FPMBand(..., modes=, mode_w=, dark_gain="profile")`：強度をモードごとの像の強度の重み付き和で計算します。明視野は同じモデルの直接光で割り、暗視野はモデルでも直接光を引いたうえで画像ごとの倍率を最小二乗で決めます（`dark_gains()` で取り出せます）。規格化の平均は窓関数の重み付きです。
- 指定しなければ従来と同じ計算です。036 での使い方は `d036/blis_df.py`（環境変数 `BLIS_DF_PC`、`BLIS_DF_PCMODES`、`BLIS_DF_GAIN`）を見てください。

## 位置の 2×2 行列（posaffine ステップ、2026-09-27 追加、32a）
平行ビーム照明・対物 FZP 走査（32a）では、レンズの位置に比例する像のずれ（ステージの倍率・軸の傾き）が残り、これを瞳が非点として肩代わりして、再構成に楕円状の癖が出ました。
`posaffine` は、このずれを瞳とは別の「位置の 2×2 行列」として当てはめ、機械的な誤差として補正します。既定では実行されません（従来と同じ動作）。

```bash
# 設定に "position_affine": {"enabled": true} があれば "all" に含まれる。明示的に指定してもよい
python run_pipeline.py --config config_32a_bf_posaffine_example.json --steps posaffine,linear,nonlinear,export,figures
```

処理（`fptrecon/posaffine.py` の `run`）：
1. 隣り合う画像（k 空間で近い `n_neighbours` 枚）の相互相関 → 各画像のずれ s_n（最小二乗、閉合誤差を報告）
2. s_n = B·(照明位置) + 高次（勾配場）+ ランダム に分解（`fit_drift`）
3. ずれを全部除いたデータで BLIS-FPM を解き、物体のピントを合わせる（`autofocus`：帯域内の対数振幅と位相の比例からのずれを最小化）。瞳の 2 次位相 − 物体の再合焦 = コントラストのピントを瞳に割り当てる（瞳と物体の 2 次位相はコントラストには同じように効き、コントラストが決めるのは和だけ。一方、像のずれ＝視差を生むのは瞳だけなので、物体のピントを合わせた状態の瞳が視差を決める）
4. その瞳が予測する視差 −∇W(k_n)/2π（`pupil_shifts_px`、符号は `selftest_sign` で確認）を残し、それ以外を画像から補正
5. 確認：位置を固定して解き直し、(瞳 − 物体の再合焦) が予測する比例ずれ − 残したずれ（`check.total_focus_drift_minus_retained_pct`）と、モデル像との残りの比例ずれ（`check.residual_linear_drift_pct`）が `warn_percent`（既定 0.1 %）未満なら `consistent: true`。ログに毎回表示されます。`check.pupil_drift_minus_retained_pct`（瞳だけの比例ずれ − 残したずれ）は参考値です（解き直した BLIS-FPM が 2 次位相の一部を物体側に移し、視差を自由な瞳の局所的な傾きで表すと大きくなります）

設定項目（`position_affine`）：`enabled`、`fov_factor`（瞳の当てはめに使う c、既定 1）、`n_neighbours`（4）、`n_pupil` / `n_joint`（BLIS-FPM の反復数）、`model_refine`（モデル像との相互相関で残りのランダムなずれを補正）、`warn_percent`、`poly_order`（瞳の多項式の次数、4）、`centre_exclude`（瞳中心の除外半径 / kc、0.25）。

`nonlinear.py` の変更：`run_nonlinear(..., free_shifts=True)` で画像ごとのずれを同時に推定できるようにしました（既定は False で従来どおり。posaffine では使っていません）。
変更前のファイルは `*.bak_20260927` に残してあります。

### フォーカスループ（2026-09-28 追加）

`position_affine` に `focus_iter`（既定 0 = 従来どおり）と `focus_tol_um`（既定 100 µm）を追加しました。`focus_iter > 0` のとき、手順 4 のあとで次をくり返します。

1. 瞳をコントラストのピント（デフォーカス＋非点のみ、`analytic_W`）に固定し、その視差を残したデータで物体だけを解く（`run_nonlinear(..., fix_pupil=True)`）
2. 物体のピントを合わせる（`autofocus`）。再合焦量が `focus_tol_um` 以上なら、それを瞳に移し（視差も計算し直して補正をやり直す）、1 に戻る

例：`python run_pipeline.py --config config_32a_bf_posaffine_focusloop_example.json --steps posaffine,linear,nonlinear,export,figures`

32a の結果：瞳 −1.53 mm → 物体の再合焦 +1.19 mm。瞳 −2.72 mm → 再合焦 +0.01 mm（2 回で収束）。誤差（misfit）は 0.5939 → 0.5952 で、ほとんど変わりません。これは、物体側のデフォーカス z が「瞳のデフォーカス z＋像の視差 λzk」と同じ画像を作り、しかも k に比例する視差は機械ずれの等方スケールと区別できないためです。データだけではどちらの解がよいか決まらず、ループは「物体のピントが合っている」解を選びます。その代わり、機械ずれの等方成分が −0.16 % 変わります。

注意：フォーカスループを使うと、手順 5 の確認（瞳を自由に動かす BLIS-FPM）で `check.ok` が false になることがあります（32a では 0.165 %）。自由な瞳は中心（|k| < 0.2 µm⁻¹）に位相の段差を持ち、動径方向の形も 2 次曲線からずれています。そのため、多項式で求めたデフォーカスは当てはめ方によって変わり、どれも固定瞳のモデル（−2.72 mm）より小さく出ます。値は、posaffine の 4 次多項式（中心 0.25 kc 除外）が −1.58 mm、回転対称の 2 次だけで当てはめると −1.85 〜 −2.09 mm（範囲の選び方による）、Zernike のバランスしたデフォーカスが −1.90 mm です。6 次多項式や中心を除かない 4 次では −0.5 〜 −1.1 mm で、さらに不安定になります。ループ自身が収束しているかは `results/position_affine.json` の `focus_loop` で確認してください。
変更前のファイルは `*.bak_20260928` に残してあります。
