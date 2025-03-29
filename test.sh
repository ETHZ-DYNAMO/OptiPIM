#!/bin/bash
# This file is used for testing of PIMopt detailed-layout process.

# --trace-output-path Path of the output trace file
# --milp-model-output: Path to store the milp model
# --memory-allocation-scheme: 
#       0. Exclusive
#       1. Combined
# --trans-coeff-method:
#       0. Comb 0
#       1. Comb 1,
#       2. Comb 2
#       3. Comb 5
#       4. Flexible, automatically select
# --target-device-type:
#       0. In-Memory-Compute (i.e. SIMDRAM)
#       1. Near-Memory-Compute (i.e. HBM-PIM)
# --number-counting-method: (default 0):
#       0. The method we proposed in the paper to estimate the unique input number
#       1. COSA's method
# --storage-method: (default 0)
#       0. No Out in a Column
#       1. No Output and Input
#       2. Store all three data arrays
# --obj-method: 0. Our cost function; 1. COSA's Cost function;
# --config-arch-path:  Path to the architecture config file
# --config-knobs-path: Path to the file contains all the weights in the obj function

./build/bin/pim-opt --data-layout-pass \
                    --trace-output-path ./Result/Group.txt \
                    --model-debug-file ./Debug/Haha.lp \
                    --trans-coeff-method 4 \
                    --target-device-type 0 \
                    --number-counting-method 0 \
                    --obj-method 0 \
                    --storage-method 0 \
                    --config-arch-path ./data/hbm_pim.json \
                    --config-knobs-path ./data/knobs.json \
                    ./nn_models/alexnet/single_layers/16banks/layerinfo.alexnet.l5.fc0.linalg.mlir
