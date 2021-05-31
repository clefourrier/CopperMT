#!/bin/bash
source $1
export WK_DIR INPUTS_DIR DATA_NAME DATA_MONO langs

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_DIR="${DIR}/neural_translation/multilingual_rnns"  # Link to "multilingual_rnns"

# ------ PARAMETERS
# INPUTS
ORIGIN_DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Link to the original data files
ORIGIN_DATA_MONO="${INPUTS_DIR}/split_data/${DATA_MONO}"  # Link to the original data files
PARAMETER_DIR="${INPUTS_DIR}/parameters"  # Contains the parameter files
# WORKING
DATA_DIR="${WK_DIR}/testing_pretraining_multilingual/data"  # Where the data will be saved
WORK_DIR="${WK_DIR}/testing_pretraining_multilingual"  # Where the models will be saved
mkdir ${WORK_DIR}

# PREPROCESSING
for seed in 0 1 2; do
    #  The monolingual data must have been extended by the bilingual data
    bash neural_translation/data_preprocess.sh -l ${langs} \
        -o "${ORIGIN_DATA_MONO}/${seed}"\
        -d "${DATA_DIR}/${seed}"

    bash neural_translation/data_preprocess.sh -l ${langs} \
        -o "${ORIGIN_DATA_DIR}/${seed}"\
        -d "${DATA_DIR}/${seed}" -f
done

# TRAINING RNN AND TRANSFORMER
for cur_seed in 0 1 2; do
    for model in "transformer" "rnn"; do
        bash neural_translation/model_train.sh  -l ${langs} \
            -d "${DATA_DIR}/${cur_seed}" \
            -w "${WORK_DIR}/${model}_full/${cur_seed}" \
            -p "${PARAMETER_DIR}/default_parameters_${model}.txt" \
            -u "${USER_DIR}" -e 5

        bash neural_translation/model_finetune.sh  -l ${langs} \
            -d "${DATA_DIR}/${cur_seed}" \
            -w "${WORK_DIR}/${model}_full/${cur_seed}" \
            -p "${PARAMETER_DIR}/default_parameters_${model}.txt" \
            -u "${USER_DIR}" -e 25  # train + finetune epochs

        bash neural_translation/checkpoint_select_best.sh \
            -l ${langs} -r ${langs} \
            -w "${WORK_DIR}/${model}_full/${cur_seed}" \
            -d "${DATA_DIR}/${cur_seed}" \
            -u "${USER_DIR}" \
            -n 1 -b 1 -f

        bash neural_translation/bleu_test_save_best.sh \
            -l ${langs} -r ${langs} \
            -w "${WORK_DIR}/${model}_full/${cur_seed}" \
            -d "${DATA_DIR}/${cur_seed}" \
            -u "${USER_DIR}" \
            -n 10 -b 10 -f
        done
    done
done