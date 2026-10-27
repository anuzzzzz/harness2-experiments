#!/usr/bin/env bash
# Commit results/<exp>/ (owned by this experiment only) plus the regenerated global table, on
# top of the latest branch head. Other experiments may publish concurrently: their results
# directories never overlap ours, and the shared README table / docs data are rebuilt from all
# summaries after every rebase, so a retry never needs a merge.
#   scripts/publish_results.sh <experiment> "<commit message>"
set -euo pipefail
EXP="$1"
MSG="$2"
BRANCH="${GITHUB_REF_NAME:-main}"
keep="$(mktemp -d)"
[ -d "results/$EXP" ] && cp -a "results/$EXP" "$keep/"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for attempt in 1 2 3 4 5 6; do
  git fetch -q origin "$BRANCH"
  git reset -q --hard "origin/$BRANCH"
  rm -rf "results/$EXP"
  mkdir -p results
  [ -d "$keep/$EXP" ] && cp -a "$keep/$EXP" "results/$EXP"
  h2x report --global-only >/dev/null
  git add -A "results/$EXP" README.md docs/data
  if git diff --cached --quiet; then
    echo "nothing new to publish"
    exit 0
  fi
  git commit -q -m "$MSG"
  if git push -q origin "HEAD:$BRANCH"; then
    echo "published results/$EXP ($(git rev-parse --short HEAD))"
    exit 0
  fi
  echo "push rejected (attempt $attempt); retrying on the new head"
  sleep $((attempt * 7))
done
echo "::error::could not publish results after 6 attempts"
exit 1
