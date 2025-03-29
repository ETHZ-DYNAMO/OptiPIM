DATA_FILE_1=validation_hbmpim.csv
DATA_FILE_2=validation_simdram.csv
OUTPUT_FILE=./Graphs/validation.pdf
NORMALIZE=false
PER_LAYER=false
LOG_TRANSFORM=false
# LAYER=traces1/resnet152/NBP/layer_144_type_0

CMD="python correlation_2.py --data_file1 $DATA_FILE_1 --data_file2 $DATA_FILE_2 --output_file $OUTPUT_FILE"

if [ $NORMALIZE = true ]; then
    CMD+=" --normalize"
fi

if [ -n "$LAYER" ]; then
    CMD+=" --layer $LAYER"
fi

if [ -n "$LOG_TRANSFORM" ]; then
    CMD+=" --log_transform"
fi

if [ $PER_LAYER = true ]; then
    CMD+=" --per_layer"
fi

echo $CMD
$CMD