# FPT パイプライン（X 線フーリエタイコグラフィの再構成）

SPring-8 で測定した X 線フーリエタイコグラフィ（FPT / FPM）のデータを、3 つの方法で再構成するコードです。

- **BLIS-FPM**（帯域制限強度スペクトル FPM、`fpt_pipeline/fptrecon`）
- **DIP**（未学習の U-Net、`dip_pipeline/dipfpm`）
- **EPRY**（`epry_pipeline/eprfpm`）

コンデンサによる斜め照明（明視野、明視野 + 暗視野）、部分コヒーレンス、平行ビーム照明 + 対物 FZP スキャン、
機械的なドリフトの補正とフォーカスループに対応しています。データ 003、018、036、32a の解析に使いました。
計算の中身と各データの手順は `README_統合パイプライン.md` にあります。

## 初めて使うとき（Mac）

### Claude Code に準備を任せる

Claude Code が入っていれば、リポジトリを取ってきて（下の 2.）、そのフォルダで Claude Code を開き、`/fpt-setup` と頼みます。
下の 1.〜5. の準備を進めます（インストールと `~/.zshrc` の変更の前には許可を求めます）。

```
git clone https://github.com/<アカウント>/fpt-pipeline.git ~/fpt-pipeline
cd ~/fpt-pipeline
claude                    # デスクトップアプリなら、~/fpt-pipeline のフォルダを開く
```

以下は、同じ準備を手で行う手順です。

### 1. 道具を用意する

```
git --version                      # なければ xcode-select --install
conda --version                    # なければ Miniforge（https://github.com/conda-forge/miniforge）を入れる
git config --global core.quotepath false   # 日本語のファイル名をそのまま表示する
```

Apple Silicon の Mac（M1〜M4）では、Miniforge の arm64 版（`Miniforge3-MacOSX-arm64.sh`）を入れます。
Homebrew が Intel 版（`/usr/local` にあり、Rosetta で動く）だと、`brew install --cask miniforge` で x86_64 版が入り、torch 2.14 が入りません。
入れたあと、`python -c "import platform; print(platform.machine())"` が `arm64` になることを確かめます。

### 2. リポジトリを取ってくる

GitHub から（URL は作成者に聞いてください）：

```
git clone https://github.com/<アカウント>/fpt-pipeline.git ~/fpt-pipeline
```

GitHub を使わずに、ファイル `fpt-pipeline.bundle` をもらった場合：

```
git clone ~/Downloads/fpt-pipeline.bundle ~/fpt-pipeline
```

リポジトリは Google Drive の中に置かないでください。同期で Git の管理ファイル（`.git/`）が壊れることがあります。
ホームフォルダの下（例 `~/fpt-pipeline`）に置きます。

### 3. Python の環境を作る

```
cd ~/fpt-pipeline
conda env create -f dip_pipeline/environment_dip.yml     # 環境 dip（Python 3.12、numpy、scipy、tifffile、matplotlib、torch）
conda activate dip
pip install python-pptx                                  # スライドを作る場合のみ
```

### 4. 生データの場所を書く

```
cp paths_local.example.sh paths_local.sh      # paths_local.sh は Git に入りません
open -e paths_local.sh                        # 自分の端末のパスに書き換える
source paths_local.sh
```

共有された Google Drive のフォルダは、Finder では「共有アイテム」に表示されます。マイドライブにショートカットを作るか、
Finder でフォルダを右クリックして、option キーを押しながら「パス名をコピー」でパスを取ります。

### 5. 動くか確かめる

```
python tests/selftest.py                                 # 数十秒。すべて OK になれば準備完了
bash tests/check_32a.sh > tests/check_32a.log 2>&1       # 32a の生データがあれば。約 15 分
tail -5 tests/check_32a.log                              # 「判定: 一致」か「ほぼ一致」になれば OK
```

`check_32a.sh` は、32a の前処理から posaffine のフォーカスループまでを実行し、作成者の解析の結果（`tests/reference/`）と比べます。
作成者と違う機種（CPU や数値計算ライブラリが違う）では、posaffine の焦点や非点が µm 単位でずれます。浮動小数点の小さな差が、反復の当てはめで増えるためです。
そのため、後段で使う量（焦点と非点、画像の補正量、misfit）が `tests/compare_ref.py` の許容内なら「ほぼ一致」と判定します。

## Claude Code で使う

ターミナルからは次のように起動します。デスクトップアプリ（Code タブ）からは、`~/fpt-pipeline` のフォルダを開きます。

```
cd ~/fpt-pipeline
claude
```

- セッションの開始時に `.claude/hooks/fpt_session_env.sh`（`.claude/settings.json` で登録）が動きます。以後のコマンドでは、
  conda 環境 `dip` と `paths_local.sh` が読み込まれます。`source paths_local.sh` と `conda activate dip` は不要です。
  - 開始時に `[FPT] 各コマンドで conda 環境 dip ... を読み込みます` と出れば準備完了です。
  - `[FPT] 準備が終わっていません` と出たら、`/fpt-setup` を使います。
  - `paths_local.sh` を書き換えたら、新しいセッションで反映されます。
- デスクトップアプリは、git worktree（`.claude/worktrees/` の下の別の作業コピー）でセッションを開くことがあります。
  その場合、出力と前の計算結果はこのフォルダと共有されません（開始時に注意が出ます）。解析は `~/fpt-pipeline` そのもので開いたセッションで行います。
- Claude Code は、開いたフォルダの `CLAUDE.md`（このリポジトリの決まり）を読み込みます。初めて開くときは、フォルダを信頼するかを聞かれます。
- 新しいデータを再構成するときの手順は、スキル `fpt-recon-workflow`（`.claude/skills/fpt-recon-workflow/`）に書いてあります。
  「`/fpt-recon-workflow` で `<生データのフォルダ>` を再構成して」のように、スキルの名前を付けて頼みます。
  名前を付けないと、各自が入れている似た説明の別のスキルが選ばれることがあります。
- 例：「32a を `d32a/runs/run_32a_main.sh` で再構成して」「036 を明視野 + 暗視野で、DIP だけ解いて」
- 実験条件（エネルギー、画素サイズ、Δr、スキャン表の各列の意味と単位など）を聞かれるので、答えてください。
  申告した値がデータと合わないときは、根拠を示して確認を求めてきます。

## Git の基本（毎日の使い方）

```
git pull                     # 最新の版を取ってくる（作業を始める前に）
git status                   # 変えたファイルを見る
git add <ファイル>           # 記録するファイルを選ぶ
git commit -m "変更の説明"   # 記録する
git push                     # GitHub に送る
```

Claude Code に「変更をコミットして」「GitHub に push して」と頼むこともできます。
生データ、再構成の出力、画像、`paths_local.sh` は `.gitignore` で除外してあるので、コミットされません。
コードを変えたときは `python tests/selftest.py` を実行してからコミットします。

## GitHub に置く（作成者が最初に 1 回）

1. GitHub のアカウントを作ります（https://github.com）。同僚の方もアカウントが必要です。
2. GitHub CLI を入れて、ログインします：`brew install gh`（Homebrew がない場合は https://cli.github.com から入れる）、`gh auth login`。
3. バンドルから取ってきたリポジトリで、次を実行します。非公開（private）のリポジトリができ、内容が送られます。

   ```
   cd ~/fpt-pipeline
   git remote remove origin                 # バンドルを指している origin を外す
   gh repo create fpt-pipeline --private --source . --push
   ```

4. GitHub のリポジトリのページで Settings → Collaborators → Add people から、同僚の方を招待します。
5. 同僚の方は、招待を承認してから `git clone` します（上の「2. リポジトリを取ってくる」）。

GitHub を使わない場合は、`fpt-pipeline.bundle` を Drive などで渡します。新しい版は、作成者が
`git bundle create fpt-pipeline.bundle --all` で作り直して渡し、受け取った側は `git pull <バンドルのパス> main` で取り込みます。

## 中身

| フォルダ・ファイル | 内容 |
|---|---|
| `fpt_pipeline/` | BLIS-FPM（前処理、較正、posaffine、線形解、非線形解、TIFF、図） |
| `dip_pipeline/` | DIP（暗視野モード、部分コヒーレンス） |
| `epry_pipeline/` | EPRY（暗視野モード、部分コヒーレンス） |
| `d018/`、`d036/`、`d32a/` | データごとのスクリプト（前処理、較正、比較、FRC・MTF、図、レポート、スライド） |
| `.claude/skills/fpt-recon-workflow/` | 新しいデータの再構成の手順（Claude Code のスキル）と、これまでの注意点・条件 |
| `.claude/skills/fpt-setup/` | 新しい Mac での初回の準備の手順（Claude Code のスキル） |
| `.claude/settings.json`、`.claude/hooks/` | セッション開始時に、環境 dip と `paths_local.sh` を各コマンドで読み込むようにするフック |
| `CLAUDE.md` | Claude Code が毎回読む、このリポジトリの決まり |
| `scripts/fpt_helpers.py` | 実験条件から λ、NA、kc、焦点深度などを計算する補助のスクリプト |
| `tests/` | `selftest.py`（簡単な確認）、`check_32a.sh`（32a での確認）、`regress_unified.py`（統合のときの回帰テストの記録） |
| `paths_local.example.sh` | 生データの場所を書くひな形 |

## 注意

- 動作を確かめたのは macOS（8 コアの CPU）だけです。今のコードは CPU で動かす前提です（GPU を使う指定はありません）。
  Windows や GPU で使うように変えるときは、ブランチを作って変え、既定の動作（CPU）を残してください。
- p（試料–対物距離 0.75 m）と CZP の焦点距離（2.4 m）は実測値ではありません。
- 生データとこれまでの成果物は、このリポジトリに入っていません。使ってよいか、データの持ち主に確認してください。
