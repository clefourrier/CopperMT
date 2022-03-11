#!/bin/bash
source $1
export WK_DIR INPUTS_DIR DATA_NAME langs_bi langs langs_shared
source ../pyenv/bin/activate

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_DIR="${DIR}/neural_translation/multilingual_rnns"  # Link to "multilingual_rnns"

# ------ PARAMETERS
# INPUTS
ORIGIN_DATA_DIR="${INPUTS_DIR}/data/shared_${DATA_NAME}"  # Link to the original data files
PARAMETER_DIR="${INPUTS_DIR}/parameters/"  # Contains the parameter files
# WORKING
DATA_DIR="${WK_DIR}/shared_bilingual/data"  # Where the data will be saved
WORK_DIR="${WK_DIR}/shared_bilingual"  # Where the models will be saved
mkdir -p ${DATA_DIR}
echo "${ORIGIN_DATA_DIR} ${PARAMETER_DIR} ${DATA_DIR} ${WORK_DIR}"

# ------ Copying PREPROCESSed data
for splits in 0.10 0.20 0.30 0.40 0.50; do
    # l for languages, o for origin data dir, d for data dir to write the files to
    # f to store fine-tuning data
    cp -r $ORIGIN_DATA_DIR/$splits/ $DATA_DIR/${splits}/
done

# ------- TRAINING RNN AND TRANSFORMER
for splits in 0.10 0.20 0.30 0.40 0.50; do
    for lang_pairs in $(echo ${langs_bi} | tr "," "\n"); do
        bash "${DIR}/neural_translation/model_train.sh" \
            -l ${lang_pairs} -a ${lang_pairs} \
            -d "${DATA_DIR}/${splits}" \
            -w "${WORK_DIR}/${lang_pairs}/${splits}" \
            -p "${PARAMETER_DIR}/default_parameters_rnn.txt" \
            -u "${USER_DIR}" -e 20

        bash "${DIR}/neural_translation/checkpoint_select_best.sh" \
            -l ${lang_pairs} -r ${lang_pairs} \
            -w "${WORK_DIR}/${lang_pairs}/${splits}" \
            -d "${DATA_DIR}/${splits}" \
            -u "${USER_DIR}" \
            -n 1 -b 1

        bash "${DIR}/neural_translation/bleu_test_save_best.sh" \
            -l ${lang_pairs} -r ${lang_pairs} \
            -w "${WORK_DIR}/${lang_pairs}/${splits}" \
            -d "${DATA_DIR}/${splits}" \
            -u "${USER_DIR}" \
            -n 10 -b 10
    done
done