#!/usr/bin/env bash
# Solves one operator on one target with OptiPIM (pim-opt + Gurobi).
# Usage (from the repo root):
#   bash group12/scripts/run_optipim.sh <C1|C2|C3|M1|M2|any name in group12/operators> <hbmpim|simdram>
# Optional: ARCH=data/xxx.json to override the arch file, BANKS=32 (label only).
# Writes   group12/mappings/<target>/<op>/optipim.trace  (+ optipim.log, .lp in Debug/)
# Appends  group12/results/solve_times.csv  (one row per op,target; rerun replaces it)
set -euo pipefail
cd "$(dirname "$0")/../.."

OP=${1:?operator id}; TARGET=${2:?hbmpim or simdram}
case "$TARGET" in
  hbmpim)  DEV=1; ARCH=${ARCH:-data/hbm_pim.json} ;;
  simdram) DEV=0; ARCH=${ARCH:-data/dram.json} ;;
  *) echo "target must be hbmpim or simdram"; exit 1 ;;
esac
MLIR=group12/operators/$OP.mlir
[ -f "$MLIR" ] || { echo "no $MLIR (run prepare_operators.sh first)"; exit 1; }
[ -x build/bin/pim-opt ] || { echo "build/bin/pim-opt not found"; exit 1; }

OUT=group12/mappings/$TARGET/$OP
CSV=group12/results/solve_times.csv
mkdir -p "$OUT" Debug group12/results
rm -f "$OUT/optipim.trace"   # never keep a stale mapping from an earlier run
# Gurobi 13 presolve wrongly reports 3x3 convs infeasible at 32 banks; Gurobi reads gurobi.env from the cwd.
if [ "${PRESOLVE:-$([ "$TARGET" = hbmpim ] && echo 0 || echo 1)}" = 0 ]; then echo "Presolve 0" > gurobi.env; trap "rm -f gurobi.env" EXIT; fi

START=$(date +%s.%N)
set +e
./build/bin/pim-opt --data-layout-pass \
    --trace-output-path "$OUT/optipim.trace" \
    --model-debug-file "Debug/${OP}_${TARGET}.lp" \
    --trans-coeff-method 4 --target-device-type $DEV \
    --number-counting-method 0 --obj-method 0 --storage-method 0 \
    --config-arch-path "./$ARCH" \
    --config-knobs-path ./data/knobs.json \
    "$MLIR" > "$OUT/optipim.log" 2>&1
RC=$?
set -e
END=$(date +%s.%N)
WALL=$(python3 -c "print(f'{$END-$START:.2f}')")

# Gurobi prints "Explored N nodes (...) in X seconds" when it finishes
GRB=$(grep -oE 'in [0-9.]+ seconds' "$OUT/optipim.log" | tail -1 | grep -oE '[0-9.]+' || true)
OBJ=$(grep -oE 'Best objective [-+0-9.e]+' "$OUT/optipim.log" | tail -1 | awk '{print $3}' || true)
if [ $RC -eq 0 ] && [ -s "$OUT/optipim.trace" ] && [ -n "$OBJ" ]; then STATUS=ok; else STATUS=fail_rc$RC; fi

[ -f "$CSV" ] || echo "operator,target,arch,wall_seconds,gurobi_seconds,objective,status" > "$CSV"
grep -v "^$OP,$TARGET," "$CSV" > "$CSV.tmp" || true; mv "$CSV.tmp" "$CSV"
echo "$OP,$TARGET,$ARCH,$WALL,$GRB,$OBJ,$STATUS" >> "$CSV"

echo "$STATUS $OP $TARGET wall=${WALL}s gurobi=${GRB:-?}s obj=${OBJ:-?}"
[ "$STATUS" = ok ] || { echo "see $OUT/optipim.log"; tail -20 "$OUT/optipim.log"; exit 1; }
