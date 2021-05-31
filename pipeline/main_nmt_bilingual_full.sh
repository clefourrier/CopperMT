#!/bin/bash
source $1
export WK_DIR INPUTS_DIR DATA_NAME langs

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_DIR="${DIR}/neural_translation/multilingual_rnns"  # Link to "multilingual_rnns"

# ------ PARAMETERS
# INPUTS
ORIGIN_DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Link to the original data files
PARAMETER_DIR="${INPUTS_DIR}/parameters/bilingual_default"  # Contains the parameter files
# WORKING
DATA_DIR="${WK_DIR}/reference_models/bilingual/data"  # Where the data will be saved
WORK_DIR="${WK_DIR}/reference_models/bilingual"  # Where the models will be saved
mkdir -p "${WK_DIR}/reference_models/bilingual"
echo "${ORIGIN_DATA_DIR} ${PARAMETER_DIR} ${DATA_DIR} ${WORK_DIR}"

# ------ PREPROCESSING
for seed in 0 1 2; do
    # l for languages, o for origin data dir, d for data dir to write the files to
    # f to store fine-tuning data
    bash "${DIR}/neural_translation/data_preprocess.sh" \
        -l "es-it,es-la,it-es,it-la,la-es,la-it" \
        -o "${ORIGIN_DATA_DIR}/${seed}"\
        -d "${DATA_DIR}/${seed}"
done

# ------- TRAINING RNN AND TRANSFORMER
for cur_seed in 0 1 2; do
    for lang_pairs in "es-it" "es-la" "it-es" "it-la" "la-es" "la-it"; do
        for model in "rnn" "transformer"; do
            bash "${DIR}/neural_translation/model_train.sh"  -l ${lang_pairs} \
                -d "${DATA_DIR}/${cur_seed}" \
                -w "${WORK_DIR}/${model}_${lang_pairs}/${cur_seed}" \
                -p "${PARAMETER_DIR}/default_parameters_${model}_${lang_pairs}.txt" \
                -u "${USER_DIR}" -e 20

            bash "${DIR}/neural_translation/checkpoint_select_best.sh" \
                -l ${lang_pairs} -r ${lang_pairs} \
                -w "${WORK_DIR}/${model}_${lang_pairs}/${cur_seed}" \
                -d "${DATA_DIR}/${cur_seed}" \
                -u "${USER_DIR}" \
                -n 1 -b 1

            bash "${DIR}/neural_translation/bleu_test_save_best.sh" \
                -l ${lang_pairs} -r ${lang_pairs} \
                -w "${WORK_DIR}/${model}_${lang_pairs}/${cur_seed}" \
                -d "${DATA_DIR}/${cur_seed}" \
                -u "${USER_DIR}" \
                -n 10 -b 10
        done
    done
done