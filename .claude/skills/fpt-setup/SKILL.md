---
name: fpt-setup
description: "FPT パイプラインを新しい Mac で使えるようにする初回の準備。conda（Miniforge arm64）と環境 dip、生データの場所を書く paths_local.sh、selftest と check_32a での確認、Claude Code での使い始め方。セッション開始時に「[FPT] 準備が終わっていません」と表示されたとき、またはユーザーが導入・セットアップ・環境づくり・動作確認を頼んだときに使う。Use for: first-time setup of this repository on a new computer, conda environment dip, paths_local.sh, selftest, check_32a."
---

# FPT パイプラインの初回の準備

このリポジトリを、新しい Mac で再構成に使える状態にする手順です。リポジトリを clone し、そのルートで Claude Code を開いたところから始めます。
各段階の結果（版、パス、判定）を確かめ、最後にまとめて報告します。

## 決まり

- **端末の設定を変える操作は、内容と影響を説明し、ユーザーの許可を得てから行います。**
  対象は、ソフトのダウンロードとインストール、`conda init`（`~/.zshrc` に追記されます）、既存の conda 環境の作り直しです。
- 生データのフォルダには書き込みません。端末に固有の絶対パスは、`paths_local.sh`（Git に入らない）にだけ書きます。
- このセッションでは、セッション開始時のフック（`.claude/hooks/fpt_session_env.sh`）がまだ効いていません。準備の途中で環境や
  `paths_local.sh` を作るためです。コマンドは `source paths_local.sh` と `conda run -n dip python ...`（または環境の python のパス）を明示して実行します。
- 長い計算（check_32a）は `nohup ... > log 2>&1 &` で起動し、ログで進み具合を確かめます。

## 1. 端末を確かめる

1. `uname -m` が `arm64` か（Apple Silicon）を確かめます。`x86_64` なら Intel の Mac です。この手順は Apple Silicon で確かめたものです（Intel の Mac は未確認）。
2. 空き容量を確かめます（`df -h ~`）。Miniforge と環境 dip で約 1.2 GB、32a の確認（check_32a）の出力で約 2.6 GB、
   再構成の出力は 1 データあたり数 GB です（2026-10-07、M4 の Mac での実測）。
3. リポジトリが Google Drive の中（`~/Library/CloudStorage/...`）に無いことを確かめます。同期で `.git/` が壊れることがあるので、中にあれば
   ホームフォルダの下（例 `~/fpt-pipeline`）に clone し直すよう勧めます。
4. `git config --global core.quotepath false`（日本語のファイル名をそのまま表示する）。

## 2. conda（Miniforge）

1. `conda` があれば、その Python が arm64 かを確かめます：`file "$(conda info --base)/bin/python"`。
   x86_64 だと、torch 2.14 の macOS 用 wheel がありません。arm64 の Miniforge を別に入れるよう、ユーザーに相談します。
2. 無ければ、Miniforge の arm64 版を入れます（許可を得てから）。
   - **Homebrew が Intel 版（`/usr/local`、Rosetta）の Mac では `brew install --cask miniforge` を使いません**（x86_64 版が入ります）。
     `brew config` の `macOS: ...-x86_64` や `Rosetta 2: true` で分かります。
   - 公式の release から `Miniforge3-MacOSX-arm64.sh` を取ってきます。sha256 を release の値と照合します。
     ```
     gh api repos/conda-forge/miniforge/releases/latest --jq '.tag_name, (.assets[] | select(.name=="Miniforge3-MacOSX-arm64.sh") | .digest, .browser_download_url)'
     curl -fsSL -o /tmp/Miniforge3-MacOSX-arm64.sh <browser_download_url>
     shasum -a 256 /tmp/Miniforge3-MacOSX-arm64.sh        # digest と一致すること
     bash /tmp/Miniforge3-MacOSX-arm64.sh -b -p ~/miniforge3
     ~/miniforge3/bin/conda init zsh                     # ~/.zshrc に追記される（許可を得てから）
     ```
     `gh` が無ければ、https://github.com/conda-forge/miniforge/releases のページの sha256 を使います。

## 3. 環境 dip

1. `conda env list` で `dip` があるかを見ます。無ければ `conda env create -f dip_pipeline/environment_dip.yml`。
   あれば作り直さず、次の確認だけ行います（作り直すときは許可を得ます）。
2. 確かめます：
   ```
   conda run -n dip python -c "import platform, torch, numpy, scipy, tifffile, matplotlib; print(platform.machine(), torch.__version__, numpy.__version__, scipy.__version__)"
   ```
   `arm64 2.14.0 ...` と出れば OK です。
3. `OMP: Error #15`（libomp.dylib が二重に読まれる）で落ちる場合：古い yml で作った環境です。OpenBLAS を pthreads 版にします。
   `conda install -n dip "libopenblas=*=*pthreads*"`。`KMP_DUPLICATE_LIB_OK=TRUE` は使いません（間違った結果を出しうる回避策）。
4. スライドを作る人だけ `conda run -n dip pip install python-pptx`。

## 4. 生データの場所（paths_local.sh）

1. `cp paths_local.example.sh paths_local.sh`（あれば上書きせず、中身を確かめます）。
2. Google Drive のルートを探します：`ls -d ~/Library/CloudStorage/GoogleDrive-*`。
3. ユーザーに、手元にあるデータ（003、018、036、32a、ほかの新しいデータ）とその場所を聞きます。
   共有されたフォルダは「共有アイテム」にあり、そのままではパスで読めません。マイドライブにショートカットを作ってもらいます。
4. 各変数のフォルダが実在し、中身があることを確かめます（例 32a：`1_sample`、`1_sample_t`、`2_direct`、`2_direct_t`）。
   持っていないデータの行は `#` でコメントにします（selftest は仮のパスで動きます）。新しいデータは `FPT<ds>_RAW` の形で足します。
5. Drive のファイルが「オンラインのみ」の場合、最初の読み込みでダウンロードが走るので時間がかかります。

## 5. 確かめる

1. `python tests/selftest.py`（環境 dip で）：すべて OK になること。
2. `python scripts/fpt_helpers.py check`：`"ok": true`。
3. 32a の生データがあれば：
   ```
   source paths_local.sh && conda activate dip && nohup bash tests/check_32a.sh > tests/check_32a.log 2>&1 &
   ```
   M4 の Mac で約 18 分です（初回は Drive からのダウンロードの時間が加わります）。終わったら `tail -12 tests/check_32a.log` を見ます。
   - 「判定: 一致」か「ほぼ一致」なら OK です。別の機種では、posaffine の焦点や非点が µm 単位でずれるので「ほぼ一致」になります。
     これは、浮動小数点の差が反復の当てはめで増えるためです。
   - 「不一致」なら、次の順で見ます。コードは変えず、数値を添えてユーザーに報告します。
     - ジオメトリの行（`A ... vs ...`、`max |x_c diff|`）が一致しているか
     - `quantities used downstream` の 4 行のうち、どれが NG か

## 6. 報告と使い始め方

報告すること：
- 入れたもの・変えたもの（Miniforge、`~/.zshrc`、環境 dip の版、`paths_local.sh` の変数）
- selftest と check_32a の結果

続けて、ユーザーに次を伝えます。
- **新しいセッションで開き直す**。セッション開始時に `[FPT] 各コマンドで conda 環境 dip ... を読み込みます` と出れば準備完了です。
  - ターミナルでは `cd <リポジトリ> && claude` で開きます。
  - デスクトップアプリでは、このフォルダそのものを開きます（worktree ではなく）。
- 再構成は「`/fpt-recon-workflow` で `<生データのフォルダ>` を再構成して」と、手順書の名前を付けて頼む。
  名前を付けないと、似た説明の別のスキルが選ばれることがあります。
- `paths_local.sh` を書き換えたら、新しいセッションで反映される。
