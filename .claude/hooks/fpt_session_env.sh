#!/bin/bash
# SessionStart hook (.claude/settings.json): every Bash command of a Claude Code session uses the conda environment
# 'dip' and the raw-data variables of paths_local.sh. The desktop app does not inherit what a terminal did with
# 'source paths_local.sh' / 'conda activate dip', so both are written to $CLAUDE_ENV_FILE, which Claude Code sources
# before each Bash command. Prints one status line for Claude and never blocks the session (exit 0).
R="${CLAUDE_PROJECT_DIR:-$PWD}"
P="$R/paths_local.sh"
note=""
common=$(git -C "$R" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
gitdir=$(git -C "$R" rev-parse --path-format=absolute --git-dir 2>/dev/null)
if [ -n "$common" ] && [ "$common" != "$gitdir" ]; then      # git worktree: paths_local.sh is in the main checkout
  main=$(dirname "$common")
  [ -f "$P" ] || P="$main/paths_local.sh"
  note=" 注意: このセッションは git worktree で動いています。出力と前の計算結果はメインのフォルダ（$main）と共有されないので、解析はメインのフォルダで開いたセッションで行ってください。"
fi
base=""
for d in "$(conda info --base 2>/dev/null)" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" \
         "$HOME/opt/anaconda3" "/opt/homebrew/Caskroom/miniforge/base"; do
  if [ -n "$d" ] && [ -x "$d/envs/dip/bin/python" ] && [ -f "$d/etc/profile.d/conda.sh" ]; then base="$d"; break; fi
done
[ -f "$P" ] || P=""
if [ -z "$base" ] || [ -z "$P" ]; then
  miss=""; [ -z "$base" ] && miss="conda 環境 dip"; [ -z "$P" ] && miss="${miss:+$miss と }paths_local.sh"
  echo "[FPT] 準備が終わっていません（$miss が見つかりません）。/fpt-setup で準備してください。$note"
fi
if [ -z "$CLAUDE_ENV_FILE" ]; then
  echo "[FPT] この Claude Code では環境を自動で読み込めません。各コマンドの前に source paths_local.sh と conda activate dip（または conda run -n dip）を付けてください。"
  exit 0
fi
if ! grep -q "# fpt_session_env" "$CLAUDE_ENV_FILE" 2>/dev/null; then     # once per session (also on resume / clear)
  echo "# fpt_session_env" >> "$CLAUDE_ENV_FILE"
  [ -n "$base" ] && printf 'source "%s/etc/profile.d/conda.sh" && conda activate dip\n' "$base" >> "$CLAUDE_ENV_FILE"
  [ -n "$P" ] && printf 'source "%s"\n' "$P" >> "$CLAUDE_ENV_FILE"
fi
if [ -n "$base" ] && [ -n "$P" ]; then
  n=0; miss=""
  for v in $(source "$P" >/dev/null 2>&1; compgen -e | grep -E '^FPT[A-Za-z0-9_]*_RAW$'); do
    n=$((n + 1)); d=$(source "$P" >/dev/null 2>&1; printf '%s' "${!v}")
    [ -d "$d" ] || miss="$miss $v"
  done
  echo "[FPT] 各コマンドで conda 環境 dip（$base/envs/dip）と paths_local.sh を読み込みます。生データの変数 $n 件${miss:+、フォルダが見つからないもの:$miss}。$note"
fi
exit 0
