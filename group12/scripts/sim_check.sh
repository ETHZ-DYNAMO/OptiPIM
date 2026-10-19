#!/usr/bin/env bash
# Replays mapping files in Ramulator2 and records memory_system_cycles.
# Quick sanity check that every mapping runs in the simulator; Member 3's
# evaluate.py does the full analytical-vs-simulator comparison.
# Usage (from the repo root): bash group12/scripts/sim_check.sh [optipim|heuristic] [C1 C2 ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
MAPPER=${1:-optipim}; shift || true
OPS=${*:-C1 C2 C3 M1 M2}
CSV=group12/results/sim_check.csv
[ -f "$CSV" ] || echo "operator,target,mapper,simulated_cycles,sim_seconds" > "$CSV"
for OP in $OPS; do
  for T in hbmpim simdram; do
    TR=group12/mappings/$T/$OP/$MAPPER.trace
    [ -f "$TR" ] || { echo "skip $OP $T (no $TR)"; continue; }
    if [ "$T" = hbmpim ]; then CFG=simulator/hbmpim_config.yaml; ALLOC=new
    else CFG=simulator/simdram_config.yaml; ALLOC=simdram; fi
    S=$(date +%s)
    CYC=$(./simulator/build/ramulator2 --config_file $CFG \
          --param Frontend.path=$TR \
          --param Frontend.PimCodeGen.alloc_method=$ALLOC 2>&1 \
          | tee "group12/mappings/$T/$OP/$MAPPER.sim.log" \
          | grep -m1 memory_system_cycles | grep -oE '[0-9]+$')
    E=$(date +%s)
    grep -v "^$OP,$T,$MAPPER," "$CSV" > "$CSV.tmp"; mv "$CSV.tmp" "$CSV"
    echo "$OP,$T,$MAPPER,${CYC:-},$((E-S))" >> "$CSV"
    echo "$OP $T $MAPPER cycles=${CYC:-FAILED} ($((E-S)) s)"
  done
done
