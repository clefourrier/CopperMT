#!/bin/bash
source $1
export WK_DIR INPUTS_DIR DATA_NAME langs_bi langs_shared
source ../pyenv/bin/activate

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_DIR="${DIR}/neural_translation/multilingual_rnns"  # Link to "multilingual_rnns"

# ------ PARAMETERS
# INPUTS
ORIGIN_DATA_DIR="${INPUTS_DIR}/data/shared_${DATA_NAME}"  # Link to the original data files
PARAMETER_DIR="${INPUTS_DIR}/parameters"  # Contains the parameter files
# WORKING
DATA_DIR="${WK_DIR}/shared_multilingual/data"  # Where the data will be saved
WORK_DIR="${WK_DIR}/shared_multilingual"  # Where the models will be saved
mkdir -p ${DATA_DIR}
echo "${ORIGIN_DATA_DIR} ${PARAMETER_DIR} ${DATA_DIR} ${WORK_DIR}"

# ------ PREPROCESSING
for splits in 0.10 0.20 0.30 0.40 0.50; do
    cp -r ${ORIGIN_DATA_DIR}/${splits} $DATA_DIR/${splits}/
done

# ------ TRAINING RNN AND TRANSFORMER
for splits in 0.10 0.20 0.30 0.40 0.50; do
    # Train rnn
    bash "${DIR}/neural_translation/model_train.sh" \
        -l ${langs_bi} -a ${langs_shared} \
        -d "${DATA_DIR}/${splits}" \
        -w "${WORK_DIR}/${splits}" \
        -p "${PARAMETER_DIR}/default_parameters_rnn.txt" \
        -u "${USER_DIR}" -e 20

    bash neural_translation/bleu_test_save_best.sh \
        -l ${langs_bi} -r ${langs_bi} \
        -w "${WORK_DIR}/${splits}" \
        -d "${DATA_DIR}/${splits}" \
        -u "${USER_DIR}" \
        -n 10 -b 10
done