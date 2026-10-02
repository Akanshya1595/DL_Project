#!/usr/bin/env bash
# Exports the per-member commit history for the Individual Contribution Log.
set -e
OUT=commit_history.txt
{
  echo "=== Commits per author ==="; git shortlog -sne --all
  echo; echo "=== Full log (hash | author | date | message) ==="
  git log --all --date=short --pretty=format:"%h | %an | %ad | %s" --shortstat
} > "$OUT"
echo "Wrote $OUT"
