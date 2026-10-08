# FPT パイプライン（X 線フーリエタイコグラフィの再構成）

SPring-8 で測定した X 線フーリエタイコグラフィ（FPT / FPM）のデータを、BLIS-FPM（`fpt_pipeline/fptrecon`）、
DIP（`dip_pipeline/dipfpm`）、EPRY（`epry_pipeline/eprfpm`）で再構成するコードです。詳しい説明は `README.md` と
`README_統合パイプライン.md` にあります。

## 決まり

- コマンドはリポジトリのルートから実行します。スクリプトはルートからの相対位置でファイルを読み書きするので、フォルダ構成を変えません。
- Python の環境は conda の `dip`（`dip_pipeline/environment_dip.yml`）です。
- 生データのフォルダは環境変数で指定します（`source paths_local.sh`。ひな形は `paths_local.example.sh`）。
  セッション開始時のフック（`.claude/hooks/fpt_session_env.sh`）が、各コマンドで `dip` と `paths_local.sh` を読み込みます。
  開始時の `[FPT]` の行で確かめ、「準備が終わっていません」と出たら `/fpt-setup` の手順で準備します。
  端末に固有の絶対パスをコードや設定に書きません。生データのフォルダには書き込みません。
- 出力（`fpt_output*`、`dip_output*`、`epry_output*`、`d*/work*` など）と画像・配列のファイルは Git に入れません（`.gitignore`）。
- 新しいデータを再構成するときは、スキル `fpt-recon-workflow`（`.claude/skills/fpt-recon-workflow/SKILL.md`）の手順に従います。
  パイプラインは再構成の条件を自動では決めません。実験条件を確認し、データを診断してから条件を決めます。
- コードを変えるときは、既定の動作を変えない形にします（新しい機能は設定か引数で有効にする）。変えたあとは
  `python tests/selftest.py` を実行し、既定の設定で変更前のコミットと結果が同じになることを確かめます。
- 長い計算（数十分以上）は `nohup ... > xxx.log 2>&1 &` でバックグラウンドで実行し、ログで進み具合を確かめます。
  DIP を止めるときは、その出力フォルダに空の `STOP` ファイルを作ります。

## 報告の決まり

- 日本語で、大学生が読める説明にします。手法の名前（BLIS-FPM、DIP、EPRY）と、もとになった論文の手法を正しく書きます。
- 表示する再構成像には帯域フィルタをかけません（分解能を目で比べるため）。
- 報告の数値は解析の JSON から書き出し、手で打ちません。
- 瞳のピントなど手法ごとに違う量は、手法ごとに書きます。
