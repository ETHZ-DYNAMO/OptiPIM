#!/usr/bin/env bash
# Solves every operator on both targets. Keeps going if one fails.
# Usage (from the repo root): bash group12/scripts/run_all.sh [C1 C2 ...]
cd "$(dirname "$0")/../.."
OPS=${*:-C1 C2 C3 M1 M2}
FAILED=""
for OP in $OPS; do
  for T in hbmpim simdram; do
    bash group12/scripts/run_optipim.sh "$OP" "$T" || FAILED="$FAILED $OP/$T"
  done
done
echo; cat group12/results/solve_times.csv
[ -z "$FAILED" ] && echo "all done" || { echo "FAILED:$FAILED"; exit 1; }
