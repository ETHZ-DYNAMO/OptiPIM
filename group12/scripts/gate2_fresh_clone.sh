#!/usr/bin/env bash
# Gate 2 check: fresh clone of the team repo, reuse the existing build
# (LLVM/pim-opt/Ramulator2 are not in git), rerun all mappings, and diff
# them against the committed ones.
# Usage: bash group12/scripts/gate2_fresh_clone.sh <git-url> [existing-repo-dir]
set -euo pipefail
URL=${1:?git url}; OLD=${2:-$HOME/OptiPIM-grp12}
NEW=$HOME/gate2_check
rm -rf "$NEW"; git clone -q "$URL" "$NEW"; cd "$NEW"
ln -s "$OLD/build" build
ln -s "$OLD/simulator/build" simulator/build
mv group12/mappings group12/mappings.committed
mv group12/results/solve_times.csv group12/results/solve_times.committed.csv 2>/dev/null || true
bash group12/scripts/run_all.sh
echo "== diff of Bound/Coeff lines (empty means identical mappings) =="
for f in $(cd group12/mappings && find . -name optipim.trace); do
  diff <(grep -E 'Bound|Coeff|Tag' group12/mappings.committed/$f) \
       <(grep -E 'Bound|Coeff|Tag' group12/mappings/$f) > /dev/null \
    && echo "same      $f" || echo "DIFFERENT $f (check its objective is equal in solve_times.csv)"
done
