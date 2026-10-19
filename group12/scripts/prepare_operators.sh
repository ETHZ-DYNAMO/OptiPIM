#!/usr/bin/env bash
# Copies the 5 agreed operators from nn_models/ into group12/operators/<ID>.mlir
# and switches them from 16 to 32 banks (same sed as the repo's create_new_testcases.py).
# Usage (from the repo root): bash group12/scripts/prepare_operators.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

declare -A SRC=(
  [C1]=nn_models/resnet50/single_layers/16banks/layerinfo.resnet50.l12.conv12.linalg.mlir
  [C2]=nn_models/resnet50/single_layers/16banks/layerinfo.resnet50.l2.conv2.linalg.mlir
  [C3]=nn_models/resnet50/single_layers/16banks/layerinfo.resnet50.l1.conv1.linalg.mlir
  [M1]=nn_models/bert/single_layers/16banks/layerinfo.bert.l0.fc0.linalg.mlir
  [M2]=nn_models/bert/single_layers/16banks/layerinfo.bert.l1.fc1.linalg.mlir
)

mkdir -p group12/operators
for ID in C1 C2 C3 M1 M2; do
  f=${SRC[$ID]}
  if [ ! -f "$f" ]; then echo "MISSING $ID: $f"; continue; fi
  sed 's/num_banks = 16/num_banks = 32/' "$f" > "group12/operators/$ID.mlir"
  if grep -q 'num_banks = 32' "group12/operators/$ID.mlir"; then
    echo "ok $ID <- $f"
  else
    echo "WARNING $ID: no 'num_banks = 32' after sed, check: grep -n num_banks $f"
  fi
done
