#!/usr/bin/env bash
# Check out harvey-labs (MIT) at the pinned commit: the evaluation code (judge prompt, document
# extraction, deliverable matching) and the draft-lpa task family used by e1-lab-lpa.
#   scripts/fetch_lab.sh [dest]
set -euo pipefail
DEST="${1:-vendor/harvey-labs}"
COMMIT=b58a28ae9fde5ee4b810e53b4fd0a1915325a35d
if [ -f "$DEST/.h2x-commit" ] && [ "$(cat "$DEST/.h2x-commit")" = "$COMMIT" ]; then
  echo "harvey-labs already at $COMMIT"; exit 0
fi
rm -rf "$DEST" && mkdir -p "$DEST" && cd "$DEST"
git init -q
git remote add origin https://github.com/harveyai/harvey-labs.git
git fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
top_files=$(git ls-tree FETCH_HEAD | awk '$2=="blob"{print $4}')
# shellcheck disable=SC2086
git checkout -q FETCH_HEAD -- $top_files evaluation utils tasks/funds-asset-management/draft-lpa
echo "$COMMIT" > .h2x-commit
echo "harvey-labs checked out at $COMMIT"
