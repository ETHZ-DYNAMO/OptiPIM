#!/bin/bash
# usage: bash run_mapping.sh <layer-file-name-without-path> <simdram|hbmpim>
# e.g.   bash run_mapping.sh alexnet.l2.conv2 simdram
LAYER=$1; DEV=$2
MLIR=$(ls ./nn_models/alexnet/single_layers/16banks/layerinfo.${LAYER}.linalg.mlir)
if [ "$DEV" = "simdram" ]; then TYPE=0; ARCH=./data/dram.json; else TYPE=1; ARCH=./data/hbm_pim.json; fi
OUT=results/member1/${LAYER}_${DEV}; mkdir -p $OUT
START=$(date +%s)
./build/bin/pim-opt --data-layout-pass \
  --trace-output-path $OUT/Group.txt --model-debug-file $OUT/model.lp \
  --trans-coeff-method 4 --target-device-type $TYPE --number-counting-method 0 \
  --obj-method 0 --storage-method 0 \
  --config-arch-path $ARCH --config-knobs-path ./data/knobs.json \
  $MLIR 2>&1 | tee $OUT/log.txt
SECS=$(( $(date +%s) - START ))
COST=$(grep -o "Real Cost: [0-9.e+-]*" $OUT/log.txt | tail -1 | awk '{print $3}')
[ -f results/member1/results.csv ] || echo "layer,device,real_cost,solver_seconds" > results/member1/results.csv
echo "${LAYER},${DEV},${COST},${SECS}" >> results/member1/results.csv
