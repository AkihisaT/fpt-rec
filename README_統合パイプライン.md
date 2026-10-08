# FPT 統合パイプライン（2026-09-28、Git 版 2026-10-01）

X 線フーリエタイコグラフィ（FPT）の再構成に、これまでのセッションで使ったコードを一つにまとめたものです。
次の計算を、この一式だけで実行できます。

| 計算 | 使ったデータ | 主に使う部分 |
|---|---|---|
| コンデンサによる斜め照明（明視野） | 003、018 | `fpt_pipeline/`、`dip_pipeline/`、`epry_pipeline/`、`d018/` |
| 斜め照明（明視野 + 暗視野） | 036 | 上記 + `d036/`（各パイプラインの暗視野モード `dark_mode`） |
| 部分コヒーレンス（照明の角度広がり） | 036 | `d036/pc/`、`fptrecon/coherence.py`、`dipfpm/coherence.py`、設定の `partial_coherence` |
| 平行ビーム照明 + 対物 FZP スキャン | 32a | `d32a/`（`d32a/runs/run_32a_main.sh`） |
| 機械的なドリフトの補正とフォーカスループ | 32a | `fpt_pipeline` の `posaffine` ステップ（設定の `position_affine`） |
| 再構成法 DIP / EPRY / BLIS-FPM | すべて | `dip_pipeline/`、`epry_pipeline/`、`fpt_pipeline/`（BLIS-FPM） |

## この一式を作った理由

2026-09-28 の時点では、これらの機能は二つの版に分かれていました。

- 036 のパッケージ（`FPT036_暗視野対応パイプライン_20260926`）：暗視野モードと部分コヒーレンスが入っているが、`posaffine` とフォーカスループは入っていない。
- `暗視野-並行ビーム照明対応pipeline`（32a で使った版）：`posaffine` とフォーカスループが入っているが、部分コヒーレンス（`coherence.py`）は入っていない。

また、32a の前処理のうち 2 か所（直接光の画像の口径食からの照明の幾何の当てはめ、明視野の 60 枚を取り出すフォルダの作成）は、
解析のときに対話的に実行しただけで、スクリプトになっていませんでした。

この一式では、036 のパッケージを土台にして、32a で変更した `fpt_pipeline` の 3 ファイルを重ねました。
`fptrecon/nonlinear.py` は両方の版で変更されていたので、元の版から両方の変更を合わせています（3 方向のマージ）。
上の 2 か所は `d32a/fit_geom_direct_32a.py` と `d32a/make_subset_32a.py` としてスクリプトにしました。
`fit_geom_direct_32a.py --check` は、解析で使った `d32a/geom_direct_v2.json` と同じ値（A、c0、R_field）を出します（照明の中心の差の最大 0.0 µm）。

## 構成

| フォルダ・ファイル | 内容 |
|---|---|
| `fpt_pipeline/` | BLIS-FPM（`fptrecon`）。前処理・較正・`posaffine`・線形解・非線形解・TIFF 出力・図。詳しくは `fpt_pipeline/README.md` |
| `dip_pipeline/` | DIP（未学習 U-Net、`dipfpm`）。暗視野モードと部分コヒーレンスに対応。`dip_pipeline/README.md` |
| `epry_pipeline/` | EPRY（`eprfpm`）。暗視野モードと部分コヒーレンスに対応。`epry_pipeline/README.md` |
| `d018/` | 4 手法の比較（`compare_018.py`、`--ds` で 018 / 036bf / 036df を指定）と BLIS-FPM の検証ツール |
| `d036/` | 036 用のスクリプト（前処理、暗視野データ、暗視野の BLIS-FPM、FRC、レポート、スライド）。手順は `d036/README_036_暗視野パイプライン.md` |
| `d036/pc/` | 部分コヒーレンスの解析 |
| `d32a/` | 32a 用のスクリプト（前処理、照明の幾何、暗視野データ、BLIS-FPM、比較、FRC・MTF、図、レポート、スライド） |
| `d32a/runs/` | 32a の解析で実際に使ったシェルスクリプトと、それを順に並べた `run_32a_main.sh` |
| `tests/` | `selftest.py`（簡単な確認）、`check_32a.sh`（32a での確認、`reference/` と比較）、統合の回帰テストの記録（`regress_unified.py`、`regress_unified.json`） |
| `.claude/skills/fpt-recon-workflow/` | 新しいデータを再構成するときの手順（実験条件の確認、データの診断、照明の方式ごとの較正、比較、報告の決まり）。Claude Code のスキル。補助の関数は `scripts/fpt_helpers.py` |
| `build_pptx.py` | スライド作成の共通関数（`d036/build_slides_036.py` が読み込みます） |

各スクリプトは、この一式のルートからの相対位置（`fpt_pipeline/`、`d32a/` など）でファイルを読み書きします。フォルダ構成は変えないでください。

## 環境

```
conda env create -f fpt_pipeline/environment.yml         # fptrecon（BLIS-FPM）と d018 / d036 / d32a のスクリプト
conda env create -f dip_pipeline/environment_dip.yml     # DIP と EPRY
pip install python-pptx                                  # スライドを作る場合のみ
```

32a の解析では、すべてを DIP の環境（`dip`）で実行しました。この環境には両方のパッケージが入っています。

## 生データの場所

生データのフォルダは、すべて環境変数で指定します（2026-10-01 の Git 版から）。`paths_local.example.sh` を
`paths_local.sh` にコピーして自分の端末のパスを書き、`source paths_local.sh` で読み込みます（`paths_local.sh` は Git に入れません）。

| データ | 環境変数 |
|---|---|
| 003 | `FPT003_RAW`（`config_003.json` の `data_dir`） |
| 018 | `FPT018_RAW`（`config_018*.json` の `data_dir`） |
| 036 | `FPT036_RAW` |
| 32a | `FPT32A_RAW` |

生データのフォルダの場所は、データの持ち主に確認してください。

fptrecon の設定の `data_dir` には `${環境変数}` や `~` を書けます。新しいデータでも同じように環境変数を使うと、端末ごとに設定を書き換えずに済みます。

## 実行の手順

作業はすべてこの一式のルートで行います。スレッド数は `OMP_NUM_THREADS` と各スクリプトの `--threads` で指定します。

### 1. コンデンサによる斜め照明・明視野（003、018）

```
cd fpt_pipeline && python run_pipeline.py --config config_003.json            # BLIS-FPM（前処理・較正・線形解・非線形解・出力・図）
cd dip_pipeline && python run_dip.py --config config_dip_003.json             # DIP（平面波 c = 0 と FOV モデル c = 1）→ 集計
cd epry_pipeline && python run_epry.py --config config_epry_003.json          # EPRY
```

018 は `config_018.json`、`config_dip_018.json`（30 反復は `_it30`、1024 画素は `_1024`）、`config_epry_018.json` を使い、
`python d018/compare_018.py --grid 512` で 4 手法を比較します。

### 2. 斜め照明・明視野 + 暗視野（036）

`d036/README_036_暗視野パイプライン.md` の「036 の再現手順」に従います。要点は次のとおりです。

1. `python d036/prep_036.py`、`python d036/make_bf_subset.py`：前処理と明視野の 9 位置の取り出し
2. `cd fpt_pipeline && python run_pipeline.py --config config_036bf.json`：明視野の較正と BLIS-FPM
3. `python d036/tools/classify_036.py`、`cd d036 && python prep_036_df.py && python noise_df.py`：暗視野のデータ（`d036/work_df/`）
4. `cd d036 && BLIS_DF_RINGS=1,3 BLIS_DF_OUT=$PWD/blis_df13 python blis_df.py -1 2 -`：暗視野を含む BLIS-FPM
5. `config_dip_036df*.json`、`config_epry_036df*.json`：DIP と EPRY の暗視野モード（`dark_mode: true`）
6. `python d018/compare_018.py --grid 512 --ds 036df --outroot d036 --unfiltered-metric --tag _df`：比較

### 3. 部分コヒーレンス（036）

暗視野の計算が済んでいることが前提です。`d036/README_036_暗視野パイプライン.md` の「部分コヒーレンスの再現手順」に従って、
`d036/pc/` のスクリプト（照明の角度分布の推定、η の選択、BLIS-FPM、DIP・EPRY、比較）を順に実行します。
DIP と EPRY は設定の `partial_coherence`（`config_dip_036pc*.json`、`config_epry_036pc*.json`）で有効になります。
BLIS-FPM では `fptrecon/coherence.py` のモードを `FPMBand` に渡します（`d036/pc/` のスクリプト）。

### 4. 平行ビーム照明 + 対物 FZP スキャン（32a）

```
export FPT32A_RAW=/path/to/32a
bash d32a/runs/run_32a_main.sh > d32a/runs/run_32a_main.log 2>&1
```

`run_32a_main.sh` は、32a のレポートの主な結果（aff2：機械的なドリフトの補正 + フォーカスループ）を作った順番どおりに、次を実行します。

1. `prep_32a_reg.py`：ダークの差し引きと、レンズ位置と検出器の値の半パルスのずれの補正（`d32a/data_reg/`）
2. `make_subset_32a.py`：明視野の 60 枚（リング 0–2）のフォルダ（`d32a/data_reg_bf/`）
3. `fit_geom_direct_32a.py`：直接光の画像の口径食から照明の幾何（`d32a/geom_direct_v2.json`）
4. `fpt_pipeline` の前処理と `make_calib_32a.py`（Δr = 250 nm、照明の波数と FOV の強さ）
5. `fpt_pipeline` の `posaffine`（`config_32a_bf_aff2.json`：機械的なドリフトの補正とフォーカスループ）
6. `prep_32a_df.py`：暗視野のデータ（リング 6–7 の 95 枚）
7. `runs/chainF2_bfaff2.sh`、`runs/chainG2_dfaff2.sh`：明視野のみと明視野 + 暗視野について、BLIS-FPM・DIP（最適反復・30 反復）・EPRY を、
   平面波と FOV モデル、512 画素と 1000 画素で解きます
8. `frc/setup_halves.py aff2` と `runs/post_aff2.sh`：半分データ、32 通りの比較、図、MTF・FRC

解析（8 コア）では、7 の二つを並行して約 2.6 時間、8 が約 20 分でした（DIP の 1000 画素は 1 反復あたり約 6.5 秒）。5 の `posaffine`（フォーカスループを含む）は約 8 分でした。

### 5. 機械的なドリフトの補正とフォーカスループ（任意のデータ）

`fpt_pipeline` の設定に `position_affine` を書くと、`posaffine` ステップが有効になります（書かなければ従来どおりです）。

```
"position_affine": {"enabled": true, "fov_factor": 1.0, "n_neighbours": 4, "n_pupil": 20, "n_joint": 150,
                    "model_refine": true, "warn_percent": 0.1, "focus_iter": 3, "focus_tol_um": 100.0}
```

- 画像のずれのうちレンズ位置に比例する部分を機械的な誤差として補正し、残りのばらつきも補正します。
- `focus_iter` が 1 以上のとき、物体側に残ったピントのずれを瞳に移して解き直します（フォーカスループ）。`focus_tol_um` 以下になれば止まります。
- 実行は `python run_pipeline.py --config <設定> --steps posaffine,linear,nonlinear,export,figures` です。
- 例：`config_32a_bf_posaffine_example.json`、`config_32a_bf_posaffine_focusloop_example.json`。説明は `fpt_pipeline/README.md` の「posaffine」の節です。

### 6. 再構成法

| 手法 | 実行 | 主な設定 |
|---|---|---|
| BLIS-FPM | `fpt_pipeline/run_pipeline.py`。暗視野を含む場合は `d036/blis_df.py`（036）、`d32a/blis_32a.py`（32a） | `fov_factors`、`nonlinear`、`position_affine` |
| DIP | `dip_pipeline/run_dip.py --steps train --fov-factors 0`（と `1`）、`--steps report` | `dip.dark_mode`、`dip.partial_coherence`、`dip.n_iter` |
| EPRY | `epry_pipeline/run_epry.py --steps train`、`--steps report` | `epry.dark_mode`、`epry.partial_coherence` |

## 既定の動作は変えていません

- `posaffine` は設定に `position_affine.enabled: true` があるときだけ実行されます。`--steps all` でも、設定がなければ飛ばします。
- フォーカスループは `focus_iter: 0`（既定）で無効です。
- `fptrecon.nonlinear.run_nonlinear` の新しい引数 `free_shifts`（各画像のずれも解く）と `fix_pupil`（瞳を固定する）は、既定で `False` です。
- 暗視野モードと部分コヒーレンスは、設定に書いたときだけ有効です。

## 統合の確認（回帰テスト）

統合した一式の計算結果が、元の二つの版と一致することを確かめました（`tests/regress_unified.json`）。

| テスト | 比較 | 最大の差 |
|---|---|---|
| T1 `run_nonlinear` の既定の動作 | 036 のパッケージ、32a の版 | 0.0、0.0 |
| T2 `fix_pupil=True`（フォーカスループ） | 32a の版 | 0.0 |
| T3 `free_shifts=True` | 32a の版 | 0.0 |
| T4 部分コヒーレンスのモード（`FPMBand`） | 036 のパッケージ | 0.0 |
| T5 DIP の暗視野モード（32a 明視野 + 暗視野、6 反復） | 解析で使った DIP | 0.0 |
| T6 EPRY の暗視野モード（32a 明視野 + 暗視野、2 周） | 解析で使った EPRY | 0.0 |

T1–T3 は 32a の明視野（3 枚に 1 枚）で短い反復（瞳 3 回・同時 5 回）、T4 は乱数のデータ、T5・T6 は FOV モデル c = 1、512 画素で比べました。
計算時間の記録（`hist_t`）は比較から除いています。T5・T6 は解析の作業フォルダの設定で実行したので、その設定は一式に入れていません。
`posaffine` の符号の自己テストも通っています。

さらに、この一式をそのままの配置で使って、32a の手順 2–5（`make_subset_32a.py` から `posaffine` のフォーカスループまで）を
前処理から実行し直し、解析の結果と比べました（`d32a/data_reg` は解析で作ったものを使用）。

| 出力 | 解析の結果との差 |
|---|---|
| 照明の幾何（A、c0、R_field） | 同じ（照明の中心の差の最大 0.0 µm） |
| 前処理した 60 枚の画像 | 最大 0.0 |
| 較正（照明の波数、FOV の強さなど） | 同じ |
| `posaffine` の結果（`position_affine.json` の 28 項目） | すべて同じ |
| フォーカスループ | 同じ経過（瞳 −1.53 → −2.72 mm、物体の再焦点 +1.19 → +0.01 mm）、最終の瞳のピント −2717.2 µm |
| 補正した画像 | 最大 0.0 |

## 統合のときに直したところ

- `fpt_pipeline/config_036bf.json` の `data_dir` が解析の作業フォルダを指していたので、`../d036/data_bf`（`d036/make_bf_subset.py` の出力）にしました。
- `build_pptx.py` の `W` が 003 の作業フォルダの絶対パスだったので、呼び出し側の `W`（なければこのファイルの場所）を使うようにしました。
- `config_dip_003*.json`、`config_epry_003*.json` は 003 の解析の作業フォルダ（`test_full`、`config_test_full.json` の出力）を読んでいたので、
  `config_003.json` の出力（`fpt_output_003/work`）を読むようにしました。二つの設定の違いは、データの場所、位置のファイル名、出力フォルダの名前だけです（再構成の設定は同じです）。
- 32a の設定の `data_dir` を、解析の作業フォルダの絶対パスから `../d32a/...` にしました。
- `run_pipeline.py --steps preprocess` だけを実行したとき、`calibration.json` がまだないと最後にエラーで止まっていたので、そこで終了するようにしました（32a では較正を `d32a/make_calib_32a.py` で作るため）。計算の結果は変わりません。
- `d32a/frc/setup_halves.py` が、BLIS-FPM の半分データに使う除外リスト（`d32a/frc/ex_*.txt`）も書くようにしました。
- `d32a/make_df_aff_32a.py` の暗視野の入力フォルダを引数で選べるようにしました（既定は `work_df_nd`、なければ `prep_32a_df.py` の出力 `work_df`。二つの違いは明視野の部分とピントの値だけで、どちらもこのスクリプトで置き換えます）。

## 注意

- `posaffine` とフォーカスループを試したのは 32a（平行ビーム照明 + 対物 FZP スキャン）だけです。コンデンサをスキャンするデータ（003、018、036）では、
  レンズ位置に比例するずれという仮定が成り立つか確かめていません。
- 暗視野の前処理（`d036/prep_036_df.py`、`d32a/prep_32a_df.py`）と暗視野を含む BLIS-FPM（`d036/blis_df.py`、`d32a/blis_32a.py`）は、
  パイプラインの設定ではなくデータごとのスクリプトです。新しいデータでは、これらを参考にしてスクリプトを用意する必要があります。
- 003 のデータは、このセッションでは読めなかったので、003 の計算は実行し直していません（コードは 036 のパッケージと同じです）。
  018 と 036 の全体の計算も、統合後には実行し直していません（上の回帰テストで一致を確かめています）。
- p（試料–対物距離 0.75 m）と CZP の焦点距離（2.4 m）は実測値ではありません。32a の照明の波数と FOV の強さは、直接光の口径食から p に依らずに決めています。
- `d036/run_*.sh`、`d32a/runs/chain*.sh` は解析で実際に使ったものです。前の計算の終了を待つループが入っています。
  単独で使うときはその行を外してください（`run_32a_main.sh` は順に実行するので、そのまま使えます）。
- `config_epry_32a*aff2*.json` の `dip_results_dir` は aff（フォーカスループの前）の DIP の出力を指しています。これは EPRY の `report` ステップが比較に読むだけで、
  EPRY の再構成（`train`）には影響しません。

## Claude を通して実行する場合

この一式は通常の Python とシェルのスクリプトなので、ご自身の端末でもそのまま実行できます。
Claude Code では、リポジトリのルートで起動すると `CLAUDE.md`（決まり）とスキル `fpt-recon-workflow` が使われます（`README.md`）。
Claude Science で使う場合は、このフォルダと生データのフォルダへのアクセスを許可し、`.claude/skills/fpt-recon-workflow/SKILL.md` の手順で、と指定してください。

新しいデータでは、パイプラインが再構成の条件を自動で決めることはありません。手順書
（実験条件の確認 → データの診断 → 較正 → 4 手法の比較 → 評価と報告）に沿って、条件を調べて決めます。
