# FPT #036 暗視野・部分コヒーレンス対応パイプライン（2026-09-26）

036（シーメンススター、30 keV、FZP Δr = 200 nm、44 照明位置 × 試料あり/なし × 6 回）の再構成に使ったコード一式です。
4 手法（DIP 最適反復、DIP 30 反復、EPRY、BLIS-FPM）を、平面波モデルと FOV モデル（c = 1）で、明視野のみ（リング 1 の 9 枚）と
明視野 + 暗視野（リング 1 + 3 の 28 枚）で解くのに使いました。部分コヒーレンス（照明の角度広がり）を入れた計算にも使いました（下の「部分コヒーレンス対応」）。

## 構成

| フォルダ | 内容 |
|---|---|
| `fpt_pipeline/` | BLIS-FPM（`fptrecon`）。前処理・較正・線形解・非線形解・TIFF 出力。`FPMBand` に暗視野用の引数を追加 |
| `dip_pipeline/` | DIP（未学習 U-Net、`dipfpm`）。設定に `dark_mode` などを追加 |
| `epry_pipeline/` | EPRY（`eprfpm`）。設定に `dark_mode` などを追加 |
| `d036/pc/` | 部分コヒーレンスの解析スクリプトとシェルスクリプト |
| `d036/` | 036 用のスクリプト（前処理、暗視野データの作成、暗視野の BLIS-FPM、FRC、レポート、スライド）と、解析で実際に使ったシェルスクリプト（`run_*.sh`） |
| `d018/` | 4 手法の比較（`compare_018.py`、`--ds 036bf / 036df` で 036 に対応）と、036 でも使った BLIS-FPM の検証ツール |
| `build_pptx.py` | スライド作成の共通関数（`d036/build_slides_036.py` が読み込みます） |

各パイプラインの詳しい説明は、それぞれの `README.md` にあります。暗視野モードは各 README の「036 暗視野モード」の節です。

## 暗視野対応で追加したもの

いずれも設定や引数で有効にしたときだけ働きます。既定の動作は変えていないので、003・018 の結果は変わりません。

| 場所 | 追加した設定・引数 | 内容 |
|---|---|---|
| `fptrecon/nonlinear.py` `FPMBand` | `dark`, `wmask`, `img_weights` | 暗視野画像は直接光で規格化しない。有効画素の重み。画像ごとの損失の重み |
| `dipfpm`（設定 `dip`） | `dark_mode`, `image_weights: "noise"`, `dark_gain: "profile"`, `exclude_images` | 暗視野画像を共通の強度単位のまま使う。雑音に応じた重み。強度係数を毎ステップの最小二乗で決める。使わない画像の指定 |
| `eprfpm`（設定 `epry`） | `dark_mode`, `exclude_images`、`EPRY(wmask=...)` | 暗視野画像は直接光で規格化しない。強度係数は最初の更新で 1 回だけ推定して固定 |
| `d018/compare_018.py` | `--ds`, `--outroot`, `--unfiltered-metric`, `--star-centre y,x`, `--tag`, `--blis a.npz,b.npz` | 036 のデータセット指定。帯域制限しない物体での misfit（暗視野の評価用）。スポーク解析の星の中心 |
| `d036/blis_df.py` | 環境変数 `BLIS_DF_RINGS`, `BLIS_DF_OUT`, `BLIS_DF_DATA`（診断用：`BLIS_DF_KSCALE3`, `BLIS_DF_PUPIL`, `BLIS_DF_LININIT`） | 暗視野を含む BLIS-FPM。使うリング、出力先、半分データ（A / B） |

暗視野のデータ（`d036/work_df/`、`prep_036_df.py` と `noise_df.py` が作ります）：
- `preprocessed.npy`（44 × 1024 × 1024）：リング 1 は明視野と同じ比画像です。リング 2・3 は試料なし画像で割らずに、(試料あり − 平滑化した試料なし) / (明視野レベル × 照明分布) としています。単位は明視野レベルで、全画像に共通です。
- `preprocessed_A.npy` / `preprocessed_B.npy`：半分平均（A = 1・3・5 回目、B = 2・4・6 回目）。
- `dfmask.npy`：画素ごとの有効重み（試料なし画像が明るい画素を除外）。
- `calibration.json`：44 枚すべての照明波数（ステージ座標からの変換）。
- `noise.json`：画像ごとの雑音分散。

## 部分コヒーレンス対応（2026-09-26 追加）

照明を、少しずつ向きの違う平面波のインコヒーレントな和（混合状態）として扱う順モデルを 3 つのパイプラインに追加しました。
各画像の強度は I_n = Σ_j w_j |F⁻¹{P(k) Ô(k − k_n − Δk_j)}|² です。物体・瞳・FOV の曲率はすべてのモードで共通です。
照明の角度分布 S は、中心（ガウス、幅 `sigma` k_c）＋すそ（割合 `eta`、2 次元 Cauchy 型、幅 `ell` k_c、指数 `beta`）です。
モードは画像ごとにまとめます（`modes`：`F` が既定で 1 画像あたり約 20 モード。`A`・`E` はリング 2 用の細かい分け方）。
設定しなければ従来のコヒーレントな計算のままです。`"eta": "1mode"` とすると、同じプログラムで照明 1 本（比較用）になります。

| 場所 | 追加した設定・引数 | 内容 |
|---|---|---|
| `fptrecon/coherence.py`（`dipfpm/coherence.py` は同じファイル） | `pc_modes(pc, kn, kc, r_p)`、`adaptive_modes`、`PC_SETS` | 設定から画像ごとの照明モード（向き Δk_j と重み w_j）を作る |
| `fptrecon/nonlinear.py` `FPMBand` | `modes`, `mode_w`, `dark_gain="profile"`、`dark_gains()` | 混合状態の順モデル。暗視野は (I − 直接光) をモデルにし、画像ごとの倍率を最小二乗で決める。規格化は窓関数で重み付けした平均 |
| `dipfpm`（設定 `dip`） | `partial_coherence`, `dark_offset: "profile"`, `pc_direct_every` | 混合状態の順モデル。暗視野は画像ごとに倍率とオフセット。直接光のキャッシュを何反復ごとに更新するか（評価のときは毎回正確に計算） |
| `eprfpm`（設定 `epry`） | `partial_coherence` | 混合状態の EPRY 更新（モードごとに強度の比で振幅を補正）。暗視野の倍率とオフセットは最初の更新で推定して固定 |
| `d018/compare_018.py` | `--dark-gain`（設定に `partial_coherence` があれば自動） | 暗視野画像に倍率を 1 つ当てはめた misfit。`--ds 036pc / 036pc1m` |
| `d036/blis_df.py` | 環境変数 `BLIS_DF_PC`（η または `1mode`）、`BLIS_DF_PCMODES`（`F`/`A`/`E`）、`BLIS_DF_GAIN` | 部分コヒーレントの BLIS-FPM。出力名に `_pc{η}{モード}` が付く |

設定の例（`dip` または `epry` の中）：`"partial_coherence": {"eta": 0.1, "modes": "F", "sigma": 0.02, "ell": 0.1, "beta": 2.0}`

### 部分コヒーレンスの再現手順

暗視野の計算（上の手順の 4 章まで）が済んでいることが前提です。

```
cd d036/pc
python fit_direct_pc.py            # 試料なし画像の直接光の曲線（図 6a）。fit_direct_pc2/3.py は c と照明角の当てはめの検証
python deconv_S.py                 # 直接光から角度分布 S を逆算（上限の目安、図 6b の破線）
python predict_pc.py               # コヒーレントな BLIS-FPM の物体で S だけ変えた予測（図 6c、predict_pc_gain.json）
python modes_convergence.py; python modes_adaptive_test.py   # モード数の確認
python synthetic_test.py           # 合成データでの確認
bash run_select_eta.sh             # η の選択（BLIS-FPM、28 枚、4 枚除外）
python make_pc_configs.py 0.1      # DIP・EPRY の設定を作る（config_dip_036pc*.json、config_epry_036pc*.json と照明 1 本の *pc1m*）
ETA=0.1 bash run_blis_pc.sh        # BLIS-FPM（28 枚・44 枚・9 枚）
bash run_dip_epry_pc.sh; bash run_dip_1m_512.sh; bash run_dip_1024_pc.sh   # DIP と EPRY
bash run_rescore_coherent.sh       # 4 章のコヒーレントな結果を同じ指標で採点し直す
GRID=512 bash run_cmp_pc.sh; GRID=1024 bash run_cmp_pc.sh                  # 4 手法の比較
python summarize_pc.py             # pc_summary.json（レポート 5 章とスライドの数値）
cd ../.. && python d036/pc/export_pc_tiffs.py   # 1024 画素の DIP（最適反復）と BLIS-FPM の TIFF（DIP 30 反復・EPRY は比較のときに出力済み）
```

- `report_pc_section.py` と `slides_pc.py` は、`d036/make_report_036.py` と `d036/build_slides_036.py` から読み込まれます（`pc_summary.json` があるときだけ）。
- 計算時間の目安（解析に使った 8 コアの Mac、2〜3 本を同時に実行）：部分コヒーレントの BLIS-FPM は 28 枚で約 25 分（4 枚を除く解は約 50 分）、44 枚で約 1 時間。照明 1 本なら約 4 分。DIP の全視野 1024 画素（FOV、150 反復）は約 45 分。

## 環境

```
conda env create -f fpt_pipeline/environment.yml         # fptrecon（BLIS-FPM、d036 のスクリプト）
conda env create -f dip_pipeline/environment_dip.yml     # dip（DIP と EPRY。EPRY は同じ環境で動きます）
pip install python-pptx                                  # スライドを作る場合のみ
```

## 036 の再現手順

作業はこの一式を展開したフォルダ（以下、ルート）で行います。生データのフォルダ（a001–a528.img）は、環境変数 `FPT036_RAW` で指定します。
指定しないと、前処理のスクリプトは止まります（2026-10-01 の Git 版から）。
スレッド数は `OMP_NUM_THREADS` と各スクリプトの `--threads` で指定します。

```
export FPT036_RAW=/path/to/036
```

**1. 前処理と明視野（リング 1、9 枚）**

1. `python d036/prep_036.py`：528 枚の分割、ホット画素の補間、試料ドリフトの補正、6 回の中央値平均を行います。出力は `d036/data/` です。
2. `python d036/make_bf_subset.py`：明視野の 9 位置を取り出します。出力は `d036/data_bf/` です。
3. `cd fpt_pipeline && python run_pipeline.py --config config_036bf.json --steps preprocess,calibrate`：較正（リング 1 のみ）を行います。
4. `cd fpt_pipeline && python run_pipeline.py --config config_036bf.json --steps linear,nonlinear,export,figures`：BLIS-FPM を解きます（c = 0 と 1）。
5. `cd dip_pipeline && python run_dip.py --config config_dip_036bf.json --steps train --fov-factors 0` と `--fov-factors 1` を実行し、最後に `--steps report` を実行します。同じことを `config_dip_036bf_it30.json`、`config_dip_036bf_1024.json`、`config_dip_036bf_1024_it30.json` でも行います。
6. `cd epry_pipeline && python run_epry.py --config config_epry_036bf.json --steps train,report` を実行します。1024 は `config_epry_036bf_1024.json` で、`--steps train` のあと `--steps report` を実行します。
7. `python d018/compare_018.py --grid 512 --ds 036bf --outroot d036` と `python d018/compare_018.py --grid 1024 --ds 036bf --outroot d036 --star-centre 290,525`：4 手法を比較します。出力は `d036/out_512/` と `d036/out_1024/` です。
8. BLIS-FPM の物体を帯域制限した検証：`FPT_CFG=fpt_pipeline/config_036bf.json FPT_WORK=fpt_pipeline/fpt_output_036bf/work FPT_OUT=d036/objband python d018/tools/blis_objband.py -1 3.5 2` を実行します（s = 0 でも実行。2 枚を除く解は末尾に `0,5`）。そのあと `d036/run_cmp_bf512.sh` で比較します。

**2. 暗視野（リング 1 + 3、28 枚）**

1. `python d036/tools/classify_036.py`：44 位置の照明波数と、明視野・境界・暗視野の分類を求めます。出力は `d036/data/illumination_036.json` です。
2. `cd d036 && python prep_036_df.py && python noise_df.py`：暗視野のデータを作ります。出力は `d036/work_df/` です。
3. `cd d036 && BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$PWD/blis_df13 python blis_df.py -1 2 -`：BLIS-FPM の FOV モデルです。第 1 引数は解のフレームでの FOV の強さ s で、s = −1 が c = 1（位相の符号 −1）、s = 0 が平面波です。第 3 引数に `0,25,34,43` を与えると、その 4 枚を除いた解になります。
4. `python d036/tools/export_blis_df.py`：BLIS-FPM の暗視野の解を、物理フレームの TIFF にします。
5. `cd dip_pipeline && python run_dip.py --config config_dip_036df.json --steps train --fov-factors 0` と `--fov-factors 1` を実行し、最後に `--steps report` を実行します。`_it30`、`_1024`、`_1024_it30` でも同じです。
6. `cd epry_pipeline && python run_epry.py --config config_epry_036df.json --steps train` のあと `--steps report` を実行します。1024 は `config_epry_036df_1024.json` です。report は DIP の結果を読むので、5 のあとに実行します。
7. `python d018/compare_018.py --grid 512 --ds 036df --outroot d036 --unfiltered-metric --tag _df` を実行します。BLIS-FPM を 4 枚除外の解にするには、`--tag _df_ho --blis d036/blis_df13/s+0_ho.npz,d036/blis_df13/s-1_ho.npz` を付けます。1024 は `--grid 1024 ... --star-centre 290,525` です（`d036/run_cmp_df1024.sh`）。
8. 半分データの FRC：`cd d036 && python make_half_workdirs.py` を実行します。次に、BLIS-FPM の半分データを `BLIS_DF_RINGS=1`（および `1,3`）、`BLIS_DF_DATA=A`（および `B`）で解きます（`d036/run_halves2.sh`）。EPRY は `run_epry.py --fpt-work-dir ../d036/work_bfA --output-dir epry_output_036bf_halfA --steps train` で解きます（B と `work_dfA/B` も同様）。最後に `cd d036 && python frc_halves.py` を実行します。

**3. レポート・スライド・まとめフォルダ**

1. `python d036/make_report_036.py`：レポートを作ります。出力は `d036/036_report.md` と `d036/036_summary.json` で、数値はすべて解析結果の JSON から取り込みます。
2. `python d036/build_slides_036.py`：スライドを作ります。出力は `d036/deck/FPT036_再構成まとめ.pptx` です。
3. `python d036/make_bundle_036.py`：Drive に置いたまとめフォルダを組み立てます。出力は `d036/bundle/` です。

**診断用（レポートの注意点や改善案の根拠）**

- `d036/tools/epry_df_diag.py`：EPRY の強度係数の方式の比較（`collapse` / `firstvisit` / `brightonly` / `r13` / `r13fix`）。
- `d036/tools/df_predict.py`：データとモデル像の比較（図 4）。
- `d036/tools/eval_df.py`：BLIS-FPM の解のリングごとの misfit。
- `d036/tools/illum_level.py`：暗視野位置の照明強度（`cd d036` で実行）。
- `d036/tools/fit_kappa_direct.py`：試料なし画像の境界からの FOV の強さの推定。
- `d018/tools/kappa_scan.py`：BLIS-FPM の misfit による FOV の強さの走査（`FPT_CFG` / `FPT_WORK` / `FPT_OUT` で 036 を指定）。
- `d036/run_k3test.sh`、`d036/run_ptest.sh`：リング 3 の照明角と瞳の半径を変えた検証。

## 注意
- `d036/run_*.sh` は解析で実際に使ったものです。前の計算の終了を待つループが入っているので、単独で使うときはその行を外してください。
- 各スクリプトは、ルートからの相対位置（`fpt_pipeline/`、`d036/` など）でファイルを読み書きします。フォルダ構成は変えないでください。
- p（試料–対物距離 0.75 m）と CZP の焦点距離（2.4 m）は実測値ではありません（`config_036bf.json` の `_comment`）。
