---
name: fpt-recon-workflow
description: "X線フーリエタイコグラフィ（FPT / FPM）の新しい測定データを、FPT統合パイプライン（BLIS-FPM・DIP・EPRY）で再構成するときの手順。実験条件の確認、データの診断（位置の整合、Δr、ステージの単位、ドリフト、明視野と暗視野の分類、部分コヒーレンス）、照明の方式（コンデンサ斜め照明／平行ビーム＋対物FZPスキャン）ごとの較正、4手法×2モデル×2格子の比較、分解能（FRC・MTF）、日本語の報告の決まり。Use for: X-ray Fourier ptychography reconstruction of a new dataset, fptrecon / dipfpm / eprfpm configs, dark-field mode, partial coherence, posaffine drift correction, focus loop, Siemens star resolution."
---

# FPT の新しいデータの再構成手順

FPT統合パイプラインは、設定に書いた条件で再構成を実行する道具です。条件を自分で決める機能は一部しかありません。
003、018、036、32a の条件は、どれもデータを調べて決めました。新しいデータでも、この手順に沿って条件を調べ、
ユーザーに確認しながら決めてください。

詳しい注意点は `references/pitfalls.md`、これまでのデータの条件は `references/datasets.md` にあります。
補助の関数はリポジトリの `scripts/fpt_helpers.py` にあります（`python scripts/fpt_helpers.py basic --energy 30 --dr 0.25` など。
Python から `sys.path.insert(0, "scripts"); from fpt_helpers import fpt_basic_numbers, fpt_k_over_kc, fpt_package_check` でも使えます）。

## 0. 準備

1. このリポジトリのルートで作業します（説明は `README.md` と `README_統合パイプライン.md`）。
   `python scripts/fpt_helpers.py check` で構成を確かめます。
2. 環境：conda 環境 `dip`（`dip_pipeline/environment_dip.yml`：Python 3.12、torch 2.14（PyPI）、tifffile、scipy、matplotlib）。
   スライドを作るときは `pip install python-pptx`。コマンドは `conda activate dip` のあとで、または `conda run -n dip python ...` で実行します。
   セッション開始時に `[FPT] 各コマンドで conda 環境 dip ... を読み込みます` と出ていれば、フックが dip と `paths_local.sh` を読み込み済みです。
   「準備が終わっていません」と出たら、先に `/fpt-setup` で準備します。
3. 生データのパスは環境変数で渡します（`paths_local.sh`、ひな形は `paths_local.example.sh`）。新しいデータも
   `FPT<ds>_RAW` のような環境変数にし、端末に固有のパスをコミットしません。生データのフォルダには書き込みません。
4. 出力（数 GB）はリポジトリの中の出力フォルダ（`.gitignore` で除外済み）に置き、コミットしません。
5. 新しいデータ用に、ルートに `d<ds>/`（前処理・較正・比較のスクリプト、`runs/*.sh`）を作ります。
   設定は各パイプラインのフォルダに `config_<ds>*.json`、`config_dip_<ds>*.json`、`config_epry_<ds>*.json` として置きます。
   いちばん近いデータの設定をコピーして書き換えます（コンデンサ照明は 018、暗視野は 036、平行ビームは 32a）。
   比較のスクリプト `d018/compare_018.py --ds <ds>` は、出力フォルダの名前 `dip_output_<ds>`、`dip_output_<ds>_it30`、
   `epry_output_<ds>` を前提にしているので、その名前に合わせます。
6. 作業の区切りごとに、スクリプトと設定を Git にコミットします（出力と生データは除く）。

## 1. 実験条件を確認する

ユーザーに次を尋ね、`fpt_basic_numbers()` で λ、NA、kc、焦点深度、κ_nom などを計算して示します。

- X 線エネルギー、試料面換算の画素サイズ、対物 FZP の Δr と直径
- 試料–対物距離 p、CZP の焦点距離（実測かどうか）
- 照明の方式（コンデンサのスキャン／平行ビーム＋対物 FZP のスキャン／その他）、ビームの大きさ
- スキャン表（CSV）の各列の意味と単位（1 パルスあたりの µm）、露光時間
- ファイルの形式（TIFF、ITEX `.img`、その他）、試料あり／なしのフォルダ、繰り返しの回数と並び順、ダークの有無
- 試料（シーメンススターなど）と、見たい構造

**ユーザーの申告値が正しいとは限りません。** 32a では Δr 200 nm と申告されましたが、データの瞳の半径は kc ≈ 2.0 µm⁻¹（Δr 250 nm）でした。
FZP Y の単位は 0.6 µm と申告されましたが、スキャンのリングが円になるのは 0.2 µm のときでした（ユーザーが確認）。
データと合わないときは、根拠（図と数値）を示してユーザーに確認します。決められないときは両方で解いて比べます。

## 2. データを診断する（再構成の前）

各項目で、図か数値を残して判断します。方法と過去の値は `references/pitfalls.md`。

1. **枚数と並び**：枚数 = 位置数 × 繰り返し × （試料あり／なし）か。並び（交互／ブロック）を確かめます。
2. **ダークとホット画素**：ダークの値（ITEX のヘッダ `ConversionFactorOffset` など）、ホット画素の割合、ザップ。
3. **試料のドリフト**：繰り返しの間で、比の画像の相互相関で確かめます（036 では最大 75 px）。
4. **位置の整合**：各画像の試料の位置（スターの中心など）が、スキャン表から予想される位置と合うか。
   ステージの丸めによる飛び（32a では半パルスのずれで 39.1 px）がないか。飛びの大きさから画素サイズも確かめられます。
5. **照明の波数と Δr**：試料なし画像の瞳の縁、または画像のスペクトルの瞳の円から kc を確かめます。
   スキャン表から計算した照明の位置が円（リング）になるかで、ステージの単位を確かめます。
6. **明視野と暗視野の分類**：`fpt_k_over_kc()` で |k|/kc を計算します。1 に近いリングは、直接光の裾や縁の縞が強く、
   モデルが合わないことがあります（036 のリング 2、|k|/kc 1.31–1.37 は除外しました）。
7. **部分コヒーレンス**：試料なし画像で、直接光が瞳の縁の外に裾を引くか（036 は引いた → 部分コヒーレンスを入れた）。
   縁のぼけが等方的で小さければ不要です（32a）。
8. **ノイズ**：繰り返しがあれば差から、1 回撮影ならスペクトルの高周波の平坦部から見積もります。

## 3. 照明の方式ごとの較正

### A. コンデンサによる斜め照明（003、018、036）

1. `cd fpt_pipeline && python run_pipeline.py --config config_<ds>.json --steps preprocess,calibrate`
2. `figures/fig1_calibration.png` で、瞳の円が各スペクトルの形（NA の縁、砂時計の形）に合っているか確かめます。
   WOTF の較正には誤った解（スケール 1.0 付近で k0 が大きい）があるので、図で確かめるまで次に進みません。
   過去のスケールは名目値の 1.15–1.32 倍でした。
3. ツイン像の符号（`twin`）：自動判定（線形解の corr(a, φ)）が弱いとき（018 は −0.11）は、根拠を示して `+1` か `−1` に固定し、
   根拠を報告に書きます（018 の根拠：003 との位相の相関、FOV 走査の最小、ピントの符号）。

### B. 平行ビーム＋対物 FZP スキャン（32a）

1. `calibrate` ステップは使えません（リング 0–1 の比の画像に FZP 0 次光の弧の縞があり、WOTF の misfit ≈ 0.8）。
2. 試料なし画像の口径食から照明の幾何を当てはめます（`d32a/fit_geom_direct_32a.py` を雛形に）。
   k_n / kc = −x_c,n / R は p に依りません。
3. `d32a/make_calib_32a.py` を雛形に `calibration.json` を作ります。
4. レンズ位置と検出器の値のずれ（丸め）を、`d32a/prep_32a_reg.py` のように補正します。
5. `posaffine`（設定の `position_affine.enabled: true`）で、レンズ位置に比例する機械的なドリフトを補正します。
   物体側にピントのずれが残れば、フォーカスループ（`focus_iter: 3`、`focus_tol_um: 100`）で瞳に移します。
   32a の一連の手順は `d32a/runs/run_32a_main.sh` です。

### C. それ以外の方式

新しい前処理か較正のスクリプトが必要です。ユーザーにそう伝え、光学系とスキャンの図と数値をもらってから作ります。
`posaffine` は 32a でしか試していません。ほかの方式で使うときは、仮定（ずれがレンズ位置に比例する）を確かめます。

## 4. 再構成（標準の比較）

ユーザーが毎回求める比較です。

- 手法：DIP（最適反復：検証用に除いた画像で選ぶ）、DIP（30 反復）、EPRY、BLIS-FPM
- モデル：平面波（c = 0）と FOV モデル（c = 1、p = 0.75 m）
- 格子：中心 512 画素と全視野（1000 または 1024 画素）
- 再構成するもの：透過像、位相像、瞳関数
- 暗視野の画像があれば：明視野のみと、明視野＋暗視野の両方

コマンド（一式のルートから）：

```
cd fpt_pipeline && python run_pipeline.py --config config_<ds>.json --steps linear,nonlinear,export,figures
cd dip_pipeline && python run_dip.py --config config_dip_<ds>.json --steps train --fov-factors 0   # と 1、最後に --steps report
cd epry_pipeline && python run_epry.py --config config_epry_<ds>.json --steps train,report
python d018/compare_018.py --grid 512 --ds <ds> --outroot d<ds> --unfiltered-metric
```

- 暗視野：DIP と EPRY は設定の `dark_mode: true`（作業フォルダに `dfmask.npy`、`noise.json` が要ります）。前処理は
  `d036/prep_036_df.py`、`d32a/prep_32a_df.py` を雛形に作ります。BLIS-FPM は `d036/blis_df.py`、`d32a/blis_32a.py`。
- 部分コヒーレンス：`d036/pc/` の手順。DIP と EPRY は設定の `partial_coherence`。
- 長い計算（数十分以上）は、`nohup bash d<ds>/runs/xxx.sh > d<ds>/runs/xxx.log 2>&1 &` のようにログに書き出してバックグラウンドで実行し、
  `tail` でログを見て進み具合を確かめます。DIP を止めるときは出力フォルダに空の `STOP` を作ります。

## 5. 評価と判断

- 主な指標：検証用に除いた画像の band misfit（0.3–3.5 µm⁻¹、`dipfpm.evaluate.band_misfit`）を、**フィルタをかけない**物体で計算します。
  帯域通過した物体での値は副次的に記録します。
- 分解能：半分データの FRC（half-bit）と、シーメンススターのスポークの変調（SNR > 3）と MTF。
  半分データは、繰り返しがあれば繰り返しで、なければリングに沿って交互に分けます。DIP の二つは異なる seed で解きます
  （同じ seed では FRC が見かけ上高くなります）。
- 合成開口の限界（kc + 最大の照明の波数）より上の FRC は、漏れが支配するので分解能として読みません。
- 瞳のピントは手法ごとに違います。図や表の「瞳 z」は手法ごとに書きます（`posaffine` の値は初期値にすぎません）。
- 解き直すたびに、数値だけでなく再構成像そのもの（前後の比較）を示します。
- 結果が物理的におかしいとき（大きな非点収差、楕円状のぼけなど）は、光学の収差と決める前に、位置の誤差やドリフトを疑います
  （32a の大きな非点収差は、機械的なドリフトを瞳が吸収したものでした）。

## 6. 報告の決まり

- 日本語で、大学生が読める説明にします。手法の名前と、もとになった論文の手法（EPRY、ePIE、rPIE、APIC など）を正しく書きます。
  プロジェクト独自の手法は **BLIS-FPM**（帯域制限強度スペクトル FPM、パッケージ名は fptrecon）と呼びます。
- 表示する再構成像には帯域フィルタをかけません（分解能を目で比べるため）。
- 報告の数値はすべて解析の JSON から書き出します。手で打ちません（`make_report_*.py` → `build_slides_*.py` → `make_bundle_*.py`）。
- 成果物は、ユーザーに確認した場所の `FPT<ds>_成果物まとめ_<日付>/` に置きます。中身はレポート（md）、
  スライド（16:9、游ゴシック）、手法ごとの TIFF、JSON、使ったスクリプトと設定です。
- 申告値と違う条件を採用したとき、データの不具合を補正したときは、その根拠をレポートに書きます。
- 解析でパイプラインのコードを変えたときは、既定の動作を変えない形にします。回帰テスト（既定の設定で元の版と差 0）を実行し、
  変更はコミットに分け、何を変えたかをコミットのメッセージに書きます（`references/pitfalls.md` の「パッケージ」）。
  `python tests/selftest.py` が通ることを確かめます。

## 7. 作業環境の注意

- fptrecon の相対パス（`data_dir`、`output_dir`）は、設定ファイルのフォルダを基準にします（カレントフォルダではありません）。
  `data_dir` には `${環境変数}` と `~` を書けます。
- 動作の確認：`python tests/selftest.py`（数十秒）。32a の生データがあれば `bash tests/check_32a.sh`（約 15 分）で、
  前処理から posaffine のフォーカスループまでを実行し、作成者の解析の結果（`tests/reference/`）と比べます。
- 計算時間の目安（8 コア CPU の Mac）：DIP 512 画素 約 1 秒／反復、1024 画素 5–18 秒／反復。EPRY 512 画素 十数秒、1024 画素 1–4 分。
  BLIS-FPM の非線形 6–12 分／モデル（1024 画素、44 枚）。
- 今のコードは CPU で動かす前提です（GPU を使う指定はありません）。
