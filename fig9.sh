# Description: This script is used to generate the results for Figure 9 in the paper.

# the result directory
RESULT_DIR=exp_results/fig9

if [ ! -d "$RESULT_DIR" ]; then
  mkdir -p "$RESULT_DIR"
fi

# build the simulator
cd simulator
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake ..
make -j
cd ../..

# Validate HBMPIM model
python validation/validation.py --device hbmpim --bin simulator/build/ramulator2
# Validate SIMDRAM model
python validation/validation.py --device simdram --bin simulator/build/ramulator2

# Plot validation results
python validation/correlation_2.py --data_file1 $RESULT_DIR/validation_hbmpim.csv --data_file2 $RESULT_DIR/validation_simdram.csv --output_file $RESULT_DIR/fig9.pdf
