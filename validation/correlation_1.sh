DATA_FILE=validation_hbmpim.csv
OUTPUT_FILE=hbmpim.pdf
NORMALIZE=true
PER_LAYER=false
# LAYER=traces1/resnet152/NBP/layer_144_type_0

CMD="python correlation_1.py --data_file $DATA_FILE --output_file $OUTPUT_FILE"

if [ $NORMALIZE = true ]; then
    CMD+=" --normalize"
fi

if [ -n "$LAYER" ]; then
    CMD+=" --layer $LAYER"
fi

if [ $PER_LAYER = true ]; then
    CMD+=" --per_layer"
fi

echo $CMD
$CMD