# dipfpm — Deep Image Prior（未学習 U-Net）による X 線フーリエタイコグラフィ再構成

添付スライド p.3 の構成をそのまま実装したパイプラインです。

```
測定画像 I_n ──► U-Net ──► 吸収 a(x), 位相 φ(x)   (O = exp(-a + iφ))
 "1" ──► MLP ──► Zernike 係数 z_j        (瞳関数 P = circ · exp(i Σ z_j Z_j))
 "1" ──► MLP ──► Shift (Δx_n, Δy_n)       (画像ごと)
 "1" ──► MLP ──► intensity c_n            (画像ごと)
            └──► Physics model ──► I_pred,n ──► Loss = (1/N) Σ_n Σ_x w(x) (c_n I_pred,n − I_meas,n)² + λ‖θ_U-Net‖²
                                              └──── backpropagation（Adam）で全ネットワークを同時最適化
```

事前学習は一切使いません。ネットワークの重みそのものが「再構成の未知数」で、U-Net の構造が
画像らしい解を選ぶ正則化（Deep Image Prior）として働きます。

## 前提

前処理（ダーク・フラット補正、ジンガー除去）と照明波数 k_n の較正は、先に作った BLIS-FPM のパイプライン `fptrecon`
パイプラインの結果をそのまま使います（比較の条件をそろえるため）。

```
cd fpt_pipeline && python run_pipeline.py --config config_003.json --steps preprocess,calibrate
```

で `work/preprocessed.npy` と `work/calibration.json` ができていれば準備完了です。

## 実行

```
conda env create -f environment_dip.yml && conda activate dip
python run_dip.py --config config_dip_003.json                               # 学習（fov_factors すべて）→ 出力・比較
python run_dip.py --config config_dip_003.json --n-iter 20                   # 動作確認
python run_dip.py --config config_dip_003.json --steps train --fov-factors 0 # 平面波モデルだけ学習
python run_dip.py --config config_dip_003.json --steps train --fov-factors 1 --pupil-free   # 瞳に自由位相マップを追加
python run_dip.py --config config_dip_003.json --steps report                # 保存済みの学習結果をすべて集計・比較
```

学習は `--fov-factors` ごとに別プロセスで並列に走らせてかまいません（`<tag>_state.npz` を
`eval_every×5` 反復ごとに上書き保存します）。途中で止めたいときは出力フォルダに `STOP` という
空ファイルを置くと、次の評価時点で結果を保存して終了します。

## 物理モデル（`dipfpm/physics.py`）

- 画像 n の強度: `I_n = | F⁻¹[ P(q + e_n) · Ô(q − k̃_n) ] |²`
  - `Ô` は `O·Q` のスペクトル（`No×No` 格子）、`k̃_n` は照明波数 k_n を格子に丸めた値。
  - 丸めの残り `k_n − k̃_n` と学習する照明補正は、瞳を `q + e_n` で評価することで**サブピクセルで厳密に**
    入れています（`|F⁻¹{P(q)Ô(q−k)}|² = |F⁻¹{P(q+e)Ô(q−k̃)}|²`, `e = k − k̃`）。
    Zernike 位相のずれは 1 次のテイラー展開（実測の最大誤差 3.3×10⁻³ rad @ |e| = 0.14 µm⁻¹。誤差は |e|² に比例し、画像シフトモードで使う端数 |e| ≤ 0.043 µm⁻¹ では約 3×10⁻⁴ rad）、瞳の縁は幅 0.25 dk のロジスティック関数。
- 瞳関数: 半径 1.02 k_c の円 × exp(i Σ z_j Z_j)。Zernike は Noll 正規化の 12 モード
  （defocus, astig ×2, coma ×2, trefoil ×2, spherical, 2nd astig ×2, quadrafoil ×2）。初期値は較正で求めた
  デフォーカス・非点から。`pupil_free: true` にすると、これに画素ごとの自由位相マップを加えます（スライドにはない拡張）。
- 視野効果（FOV）: `fov_factor c ≠ 0` のとき `Q = exp(iπκ|x − x_axis|²)`, `κ = c/(λp)`（p = 0.75 m）。
  フラット画像にも同じ照明の口径食が入っているので、`I_n / I_n[Q のみ]` で規格化します（BLIS-FPM と同じ）。
- Shift: `shift_mode` で意味を選べます。
  - `image`（既定）: 画像ごとの横ずれ（試料・光学系のドリフト）。単位はカメラ画素。
  - `illumination`: 画像ごとの照明波数の補正（LED 位置補正に相当）。単位はスペクトル画素 dk。
  いずれも `shift_max` で tanh により上限を付け、学習画像の平均は 0 に固定します（全画像共通のずれは物体の平行移動と区別できないため）。
- intensity: `c_n = exp(MLP 出力)`。

## 学習と早期終了（`dipfpm/train.py`）

- 画像: BLIS-FPM と同じ 36 枚（|k|/k_c ≤ 0.97）。そのうち `val_every` 枚おき（既定 9 → 4 枚）を**検証用に外し**、U-Net・Zernike には使いません。
- 検証画像の Shift と c_n だけは、物体と瞳を固定したまま別の MLP で合わせます（外した画像でも位置ずれ・強度は未知なので）。
- 評価指標: 強度 MSE に加え、データに信号がある帯域 0.3–3.5 µm⁻¹ だけのスペクトル誤差（band misfit）。
  `select_by: "val_band"`（既定）で検証画像の band misfit を基準にし、「最小値の (1 + `select_tol`) 倍以内に入った
  最初の評価点」の結果を採用します（既定 `select_tol` = 0.01。検証曲線が平らなときに、より早い＝過学習の少ない解を選ぶため）。
  最終反復の結果も `<tag>_state.npz` の `final_*` に残します。
- 損失の窓 `loss_tukey`: 切り出し端の不連続を抑える Tukey 窓（スライドの損失に重み w(x) を付けたもの）。

## 出力（`output_dir`）

| ファイル | 内容 |
|---|---|
| `<tag>_transmission.tif`, `<tag>_phase_rad.tif` | 物理フレーム（双子像を補正済み）、0.3–3.5 µm⁻¹ 帯域に制限した透過率・位相 |
| `<tag>_*_unfiltered.tif` | 帯域制限なし |
| `<tag>_pupil_phase_rad.tif` | 瞳位相（物理フレーム、k_c 円内） |
| `<tag>_state.npz` | 採用反復（best_*）と最終反復（final_*）の a, φ, Zernike, Shift, c_n, 学習履歴 |
| `dip_summary.json` | 採用反復、各種 misfit、Zernike、Shift、c_n、BLIS-FPM との相関 |
| `fig_dip_training.png` | 学習曲線 |
| `fig_dip_comparison.png` | DIP と BLIS-FPM の比較：画像、FRC、Zernike、共通 misfit |

`tag` は `DIP_planewave`（c = 0）, `DIP_FOV_c1`（c = 1）など。`--pupil-free` で `_freepupil`, `--no-shift` で `_noshift` が付きます。

## BLIS-FPM との比較の方法

`previous_native` に指定した BLIS-FPM（fptrecon）の非線形解（未フィルタの a, φ と自由瞳位相 W）を同じ切り出し・同じ格子・
同じ指標で評価します。比較指標 band misfit は

```
Σ_n Σ_{q∈band} |F{w·(I_n/⟨I_n⟩ − 1)} − F{w·(I_meas,n − 1)}|²  /  Σ_n Σ_{q∈band} |F{w·(I_meas,n − 1)}|²
```

（どちらの手法も物体は 0.3–3.5 µm⁻¹ に帯域制限してから評価）。BLIS-FPM は 36 枚すべてで最適化されているので、
検証画像の値は DIP にとってだけ「未使用データでの誤差」です。

## 主な設定（`config_dip_003.json` の `dip`）

| キー | 既定 | 意味 |
|---|---|---|
| `crop_px`, `crop_center` | 512, 中心 | 再構成する切り出し（CPU のため 16.3 µm 角） |
| `unet_px`, `unet_base`, `unet_depth` | 256, 16, 4 | U-Net の格子と幅・深さ（出力は No 格子へフーリエ補間） |
| `absorption_scale`, `phase_scale` | 0.05, 0.5 | U-Net 出力 → a, φ の係数 |
| `zernike_scale`, `shift_scale_px`, `intensity_scale`, `shift_max` | 0.3, 1, 0.1, 5 | MLP 出力の係数・上限 |
| `lr_unet`, `lr_mlp`, `l2_weight` | 1e-3, 1e-3, 1e-4 | Adam 学習率、U-Net 重みの L2 |
| `loss` | `intensity` | `amplitude`（√I の差）も選択可 |
| `n_iter`, `eval_every`, `select_by`, `select_tol` | 600, 10, `val_band`, 0.01 | 反復数、評価間隔、早期終了の基準と許容幅 |
| `shift_mode`, `shift_max` | `image`, 5 | Shift の意味（画像の横ずれ / 照明波数補正）と上限 |
| `fov_factors` | [0, 1] | 視野効果 c（0 = 平面波） |
| `pupil_free`, `lr_pupil_free`, `pupil_free_l2` | false, 1e-3, 0 | 自由瞳位相の拡張 |
| `threads` | 8 | CPU スレッド数 |

## 1024 格子での計算（`config_dip_003_1024.json`）

`crop_px = 1024`、`unet_px = 1024` にすると、1024×1024 画素（32.7 µm 角）を U-Net もカメラ画素のまま（縮小なし）で計算します。
物体格子は No = 1024、画像パッチは M = 672 です。

- 1 反復：8 スレッドで約 4.7 s（単体の計測）。2 本並列（4 スレッドずつ）では約 7–10 s。メモリ最大約 5.5 GB。
- FOV モデルでは、切り出しが大きいと視野端でモデル上の直接光が瞳の外に出る画像があります（|k|/k_c ≥ 0.9 の画像で画素の最大 22 %）。
  この領域はフラット補正後のモデルで表せないので、`mask_vignetted: true`（既定）で損失から除きます。
  評価指標も `eval_vignetting_mask: true` にすると、全手法で同じ画素を除いて比べます。

## 分解能の評価（`tools/`）

各位置の 2 回測定（repeat 1, 2）を別々に前処理し、独立に再構成して FRC を取ります。

```
python tools/make_half_datasets.py --config ../fpt_pipeline/config_003.json --out ../fpt_pipeline/half_work
python run_dip.py --config config_half_A.json --steps train            # fpt_work_dir = half_work/A, seed 0
python run_dip.py --config config_half_B.json --steps train            # fpt_work_dir = half_work/B, seed 1（必ず別のシード）
python tools/frc_half.py --dip <A>/DIP_planewave_state.npz <B>/DIP_planewave_state.npz
python tools/run_prev_crop.py half_work/A prev_A.npz 1 && python tools/run_prev_crop.py half_work/B prev_B.npz 1
python tools/frc_half.py --prev prev_A.npz prev_B.npz                   # BLIS-FPM（同じ切り出し、初期値ゼロ）
```

2 つの DIP を同じシードで走らせると、未学習ネットワークが両方に同じ高周波構造を作るので、FRC が過大になります。

## 制限

- CPU 実行（GPU なし）のため、全視野 1500×1500 ではなく中央 512×512（16.3 µm 角）で再構成しています。
  1 反復は 8 スレッドで約 1.4 s（3 本並列時は約 3–4 s）。
- 検証画像が 4 枚と少ないので、早期終了の位置には揺らぎがあります。

## 018 での使い方（2026-09-26 追加）
- `config_dip_018.json`（中央 512 画素、600 反復）、`config_dip_018_it30.json`（同じ設定で 30 反復）、`config_dip_018_1024.json`（全視野、300 反復）、`config_dip_018_1024_it30.json`。
- `fpt_work_dir` に BLIS-FPM（fptrecon）の作業フォルダを指定し、前処理・較正をそのまま使います。
- `dipfpm/evaluate.py`：学習曲線と比較図の凡例の画像枚数（従来は 32 / 4 の固定値）を、実際の枚数（018 では 39 / 5）にしました。

## 036 暗視野モード（2026-09-26 追加）
設定の `dip` に次の項目を追加しました。書かなければ従来の動作で、003・018 の結果は変わりません（例：`config_dip_036df.json`）。

| 項目 | 値 | 内容 |
|---|---|---|
| `dark_mode` | `true` / `false`（既定） | \|k_n\| > `pupil_radius`·k_c の画像を暗視野として扱います。暗視野画像は画像ごとに平均 1 に規格化せず、明視野レベルを単位とする共通の強度のまま使います。`fpt_work_dir/dfmask.npy` を有効画素の重みとして読みます。U-Net の入力は画像ごとに標準化します。 |
| `image_weights` | `"noise"` | 損失の画像重みを λ_n = σ²(明視野の中央値) / σ²_n とします（`fpt_work_dir/noise.json` の `sigma2_512` / `sigma2_1024`）。明視野画像の重みは 1 です。 |
| `dark_gain` | `"profile"` | 暗視野画像の強度係数を、毎ステップの最小二乗 c_n = Σ w I_pred I_meas / Σ w I_pred² で決めます。書かないと、従来どおり MLP で学習する係数になります（036 では係数が 1 からほとんど動きませんでした。その版の結果は `*_v1gain`）。 |
| `exclude_images` | 画像番号のリスト | 学習にも検証にも使わない画像です（036 ではリング 2 の 9–24）。 |

- `dipfpm/evaluate.py`：`dark_mode` では、misfit の計算に同じ有効画素の重み（`WMASK`）を使います。学習曲線の図の縦軸の範囲は、データから決めるようにしました。
- 036 の設定：`config_dip_036bf*.json`（明視野 9 枚）、`config_dip_036df*.json`（リング 1 + 3 の 28 枚）。`_1024` は全視野、`_it30` は 30 反復です。

## 036 部分コヒーレンス（2026-09-26 追加）

設定 `dip` に次を加えると、照明を混合状態として扱います（`d036/pc/make_pc_configs.py` が作る `config_dip_036pc*.json`）。

| キー | 例 | 内容 |
|---|---|---|
| `partial_coherence` | `"partial_coherence": {"eta": 0.1, "modes": "F", "sigma": 0.02, "ell": 0.1, "beta": 2.0}` | 照明の角度分布とモードのまとめ方。`{"eta": "1mode"}` で照明 1 本 |
| `dark_offset` | `"profile"` | 暗視野画像ごとに倍率とオフセットを最小二乗で決める（`dark_gain: "profile"` と併用） |
| `pc_direct_every` | `5` | 学習中の直接光のキャッシュを何反復ごとに更新するか（検証誤差の計算では毎回正確に計算） |

- `physics.FPMPhysics(..., modes=, mode_w=)` が混合状態の順モデルです（`dipfpm/coherence.py` は `fptrecon/coherence.py` と同じファイル）。
- 部分コヒーレンスの設定があると、検証誤差（最適反復の選択）も暗視野に倍率を当てはめた指標になります。
