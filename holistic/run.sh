# A sample script if you just want to run one case of the holistic model solving
# usage: sh run.sh

MODEL_NAME=alexnet
MODEL_FILE="../nn_models/${MODEL_NAME}/layer_params.csv"
ARCH=simdram # simdram, hbm
N_BANKS=256 # 256, 512, 1024 for 8GB, 16GB, 32GB
LAYER_GROUPING=true # true for layer grouping, false for even
RECURRENT=false
DATABASE_FILE=../exp_results/fig14/all_layer_results.json # need to generate this file first

CMD="python calCycle.py \
     --database_file ${DATABASE_FILE} \
     --model_file ${MODEL_FILE} \
     --model_name ${MODEL_NAME} \
     --arch ${ARCH} \
     --n_banks ${N_BANKS}"

if [ ${LAYER_GROUPING} = true ]; then
    CMD="${CMD} --layer_grouping"
fi

if [ ${RECURRENT} = true ]; then
    CMD="${CMD} --recurrent"
fi

echo ${CMD}
${CMD}