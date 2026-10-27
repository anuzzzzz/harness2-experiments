#!/usr/bin/env bash
# Check out tau2-bench v2.0.0 (MIT) with only what the airline and banking_knowledge text runs
# need (source, those domains, the English language pack); skips ~800 MB of results/voice data.
#   scripts/fetch_tau2.sh [dest]        then: pip install -e "<dest>[knowledge]"
set -euo pipefail
DEST="${1:-vendor/tau2-bench}"
COMMIT=989d3285ae669e49503324bdb9981bf14fccedeb
if [ -f "$DEST/.h2x-commit" ] && [ "$(cat "$DEST/.h2x-commit")" = "$COMMIT" ]; then
  echo "tau2-bench already at $COMMIT"; exit 0
fi
rm -rf "$DEST" && mkdir -p "$DEST" && cd "$DEST"
git init -q
git remote add origin https://github.com/sierra-research/tau2-bench.git
git fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
top_files=$(git ls-tree FETCH_HEAD | awk '$2=="blob"{print $4}')
tau2_dirs=$(git ls-tree --name-only FETCH_HEAD data/tau2/ \
  | grep -v -E '^data/tau2/(results|multilingual|multilingual_extensions|domains)$' || true)
ml_files=$(git ls-tree FETCH_HEAD data/tau2/multilingual/ | awk '$2=="blob"{print $4}')
# shellcheck disable=SC2086
git checkout -q FETCH_HEAD -- $top_files src $tau2_dirs $ml_files \
  data/tau2/multilingual/en \
  data/tau2/domains/airline data/tau2/domains/banking_knowledge data/tau2/domains/mock
echo "$COMMIT" > .h2x-commit
du -sh . | cut -f1 | xargs echo "tau2-bench checked out:"
