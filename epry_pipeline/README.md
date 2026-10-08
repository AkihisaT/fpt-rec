# eprfpm — EPRY 法による X 線フーリエタイコグラフィ再構成

EPRY（embedded pupil function recovery：X. Ou, G. Zheng, C. Yang, Opt. Express 22(5), 4960–4972 (2014)）を実装した
パイプラインです。DIP（`dip_pipeline`）や BLIS-FPM（`fptrecon`）と比べられるように、次の 4 点を共通にしています。

- 前処理
- 照明波数の較正
- 切り出し範囲と使う画像（36 枚のうち 4 枚を検証用に外す）
- 評価指標（帯域内 misfit）

## アルゴリズム（`eprfpm/core.py`）

照明波数 |k_n| の小さい順に 1 枚ずつ、次の更新を繰り返します（1 周 = 1 sweep）。

```
Φ_n  = P_n · S_n                                   S_n: -k_n に対応する物体スペクトルの一片, P_n: 瞳
ψ_n  = F⁻¹{Φ_n}
ψ'_n = ψ_n + w·(A_n ψ_n/|ψ_n| − ψ_n)               振幅の置き換え, A_n = sqrt(g_n · I_Q,n · I_meas,n)
ΔΦ   = F{ψ'_n} − Φ_n
S_n += α · conj(P_n)/max|P_n|² · ΔΦ                物体スペクトルの更新
P   += β · conj(S_n)/max|S_n|² · ΔΦ                瞳の更新（EPRY）
```

順モデルの約束事は `dipfpm.physics.FPMPhysics` と同じです。

- ソルバー座標系を使います。
- 照明波数の端数は、瞳をサブピクセル（双線形）でずらして入れます。
- FOV モデルでは二次位相 Q を掛け、直接光像 I_Q で規格化します。

原論文の EPRY に加えた点（どれも設定で切り替えられます）：

| 追加 | 設定 | 理由 |
|---|---|---|
| 適応ステップ幅（Zuo, Sun, Chen, Opt. Express 24(18), 20724 (2016)） | `adaptive_step`, `step_eps`, `alpha_min` | 振幅誤差の減少が 1 sweep で `step_eps` 未満なら α, β を半分にする。一定ステップ α = β = 1 では、このデータでは収束しなかった（帯域内 misfit が 1.06 前後のまま） |
| 画像ごとの強度係数 g_n | `intensity_correction` | 各 sweep 後に最小二乗で推定（平均 1） |
| サブピクセルの照明位置 | 常に有効 | DIP・BLIS-FPM と同じ順モデル |
| Tukey 窓付きの振幅置き換え | `replace_tukey` | 切り出し端の不連続が伝わるのを抑える |
| 検証画像による sweep の選択 | `select_by`, `select_tol` | DIP と同じ規則（検証 misfit が最小値の 1 % 以内に入った最初の sweep） |
| 瞳更新の正規化 | `pupil_gamma` | 0 = EPRY（conj(S)/max\|S\|²）。> 0 にすると rPIE 型（(1−γ)\|S\|² + γ max\|S\|²）になる |
| 口径食マスク | `mask_vignetted`, `eval_vignetting_mask` | FOV モデルで、モデル上の直接光が瞳の外に出る画素を置き換え・評価から除く（1024 格子で使用） |
| 初期値の乱数 | `init_noise`, `seed` | 初期値に依存しないかを確かめる試験用 |

## 実行

```
# 前提：fpt_pipeline（preprocess, calibrate）と dip_pipeline がある
python run_epry.py --config config_epry_003.json                   # 512 画素：学習 → 出力 → DIP・BLIS-FPM と比較
python run_epry.py --config config_epry_003_1024.json              # 1024 画素
python run_epry.py --config config_epry_003.json --steps report    # 保存済みの結果から比較だけやり直す
python run_epry.py --config config_epry_003.json --steps train --fpt-work-dir ../fpt_pipeline/half_work/A --output-dir out_A
python ../dip_pipeline/tools/frc_half.py --dip out_A/EPRY_planewave_state.npz out_B/EPRY_planewave_state.npz
```

計算時間（この Mac、8 スレッド）：512 画素で 1 モデル約 13 s（17 sweep）、1024 画素で約 55–70 s でした。

## 出力（`output_dir`）

| ファイル | 内容 |
|---|---|
| `<tag>_transmission.tif`, `<tag>_phase_rad.tif` | 物理フレーム、0.3–3.5 µm⁻¹ に帯域制限（`*_unfiltered.tif` は制限なし） |
| `<tag>_pupil_phase_rad.tif`, `<tag>_pupil_amplitude.tif` | 瞳（物理フレーム） |
| `<tag>_state.npz` | 選んだ sweep（best_*）と最終 sweep（final_*）の a, φ, P, g, Zernike、履歴 |
| `epry_summary.json` | 各手法の帯域内 misfit（学習画像・検証画像）、Zernike、位相相関、FRC |
| `fig_epry_comparison.png` | EPRY・DIP・BLIS-FPM を同じ切り出し・同じ指標で比べた図 |
| `fig_epry_sweeps.png`, `fig_epry_pupil.png` | sweep ごとの misfit とステップ幅、瞳 |

`tag` は `EPRY_planewave`（c = 0）と `EPRY_FOV_c1`（c = 1）です。

## 018 での使い方（2026-09-26 追加）
- `config_epry_018.json`（中央 512 画素）、`config_epry_018_1024.json`（全視野）。データの準備は dip_pipeline の設定を使います。
- `eprfpm/report.py`：sweep 図の凡例の画像枚数を実際の枚数にしました。

## 036 暗視野モード（2026-09-26 追加）
- 設定の `epry` に `dark_mode`（既定 `false`）と `exclude_images` を追加しました。`dark_mode` のときは、データの準備（dip_pipeline の読み込み）で `dfmask.npy` を読み、`EPRY(..., wmask=...)` に渡します。
- `eprfpm/core.py`：`wmask` を渡すと、\|k_n\| > `pupil_radius`·k_c の画像を暗視野として扱います。直接光による規格化はしません（I_Q := 1）。コントラストは有効画素の平均のまわりで取ります。
- 暗視野画像の強度係数は、その画像を最初に更新するときに 1 回だけ最小二乗で推定し、以後は固定します。従来のように掃引ごとに推定し直すと、係数が 0 に縮む解（0.83 → 0.024）に落ちたためです（`d036/tools/epry_df_diag.py` の `collapse` と `firstvisit` の比較）。
- `wmask` を渡さなければ従来と同じ動作で、003・018 の結果は変わりません。
- 036 の設定：`config_epry_036bf*.json`（明視野 9 枚）、`config_epry_036df*.json`（リング 1 + 3 の 28 枚）。

## 036 部分コヒーレンス（2026-09-26 追加）

設定 `epry` に `"partial_coherence": "partial_coherence": {"eta": 0.1, "modes": "F", "sigma": 0.02, "ell": 0.1, "beta": 2.0}` を加えると、混合状態の EPRY になります（`config_epry_036pc*.json`）。

- 更新：各照明モードの像の振幅を、測定強度とモデル強度（全モードの和）の比で補正し、物体と瞳をモードの重みで平均して更新します。
- 暗視野画像の倍率とオフセットは、最初に更新するときに推定して固定します。
- 検証誤差は暗視野に倍率を当てはめた指標です。036 では、照明 1 本でも部分コヒーレントでも、検証誤差が最小になるのは 2〜4 回目の掃引でした。
