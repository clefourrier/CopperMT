#!/bin/bash
source $1
export WK_DIR INPUTS_DIR DATA_NAME DATA_MONO_BT_ORIG langs

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_DIR="${DIR}/neural_translation/multilingual_rnns"  # Link to "multilingual_rnns"

# ------ PARAMETERS
# WORKING
DATA_DIR="${WK_DIR}/testing_backtrad_multilingual/data"  # Where the data will be saved
WORK_DIR="${WK_DIR}/testing_backtrad_multilingual"  # Where the models will be saved
WORK_DIR_REFERENCE="${WK_DIR}/reference_models/multilingual"  # Where the models must be
# INPUTS
# - Backtranslated data generation folders
ORIGIN_DATA_RAW="${INPUTS_DIR}/raw_data/${DATA_MONO_BT_ORIG}"  # Link to the original data files
DATA_MONO_DIR="${WORK_DIR}/data_mono"  # Where the data will be read by the backtranslation generation models
RAW_DATA_BACKTRAD="${INPUTS_DIR}/raw_data/backtrad/multilingual_NMT"  # Where the raw backtranslated data will be saved afterr extraction
SPLIT_DATA_BACKTRAD="${INPUTS_DIR}/split_data/backtrad/multilingual_NMT"  # Where the raw backtranslated data will be saved afterr extraction
# - Training models
ORIG_DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Link to the original data files
PARAMETER_DIR="${INPUTS_DIR}/parameters"  # Contains the parameter files

UNIQ_LANGS=$(for l in $(echo $(echo ${langs} | tr "," "\n") | tr "-" "\n"); do echo $l; done | sort | uniq)
echo ${UNIQ_LANGS}

# Generate backtranslated data
for seed in 0 1 2; do
    # Preprocessing monolingual data
    # -- We use the same embedding as for the original models
    mkdir -p "${DATA_MONO_DIR}/${seed}/data-bin/"
    for U_LANG in ${UNIQ_LANGS}; do
        cp "${WORK_DIR_REFERENCE}/data/${seed}/data-bin/dict.${U_LANG}.txt" \
           "${DATA_MONO_DIR}/${seed}/data-bin/dict.${U_LANG}.txt"
    done
    bash neural_translation/data_preprocess.sh -l ${langs} \
        -o "${ORIGIN_DATA_RAW}"\
        -d "${DATA_MONO_DIR}/${seed}" -r

    # Predicting backtranslated datasets and extracting backtranslated information
    for model in "transformer" "rnn"; do
        # 1) Copy the reference models
        mkdir -p "${WORK_DIR}/${model}_reference/${seed}/checkpoints"
        cp -a "${WORK_DIR_REFERENCE}/${model}_full/${seed}/checkpoints" \
           "${WORK_DIR}/${model}_reference/${seed}"

        # 2) Predict 10 best
        bash neural_translation/bleu_test_save_best.sh \
            -l ${langs} -r ${langs} \
            -w "${WORK_DIR}/${model}_reference/${seed}" \
            -d "${DATA_MONO_DIR}/${seed}" \
            -u "${USER_DIR}" \
            -n 10 -b 10

        mkdir -p "${RAW_DATA_BACKTRAD}/${model}/${seed}"

        python data/dataraw_extract_from_backtranslated.py \
            ${langs} "${ORIGIN_DATA_RAW}" \
            "${WORK_DIR}/${model}_reference/${seed}" \
            "${RAW_DATA_BACKTRAD}/${model}/${seed}"

        bash data/dataraw_shuffle_split.sh \
            -l "${langs}" -r "${RAW_DATA_BACKTRAD}/${model}/${seed}" \
            -d "${SPLIT_DATA_BACKTRAD}/${model}" \
            -s "${seed}" -t 90 -v 99

        bash data/data_concatenate.sh \
            -d "${ORIG_DATA_DIR}/${seed}" \
            -t "${SPLIT_DATA_BACKTRAD}/${model}/${seed}"
    done
done


# PREPROCESSING
# Here, the data is model specific, and has been generated
# by the corresponding reference model
for seed in 0 1 2; do
    for model in "transformer" "rnn"; do
        bash neural_translation/data_preprocess.sh -l ${langs} \
            -o "${SPLIT_DATA_BACKTRAD}/${model}/${seed}"\
            -d "${DATA_DIR}/${model}/${seed}"      
                
        bash neural_translation/data_preprocess.sh -l ${langs} \
            -o "${ORIG_DATA_DIR}/${seed}"\
            -d "${DATA_DIR}/${model}/${seed}" -f
    done

done

# TRAINING RNN AND TRANSFORMER
for seed in 0 1 2; do
    for model in "transformer" "rnn"; do
        bash neural_translation/model_train.sh  -l ${langs} \
            -d "${DATA_DIR}/${model}/${seed}" \
            -w "${WORK_DIR}/${model}/${seed}" \
            -p "${PARAMETER_DIR}/default_parameters_${model}.txt" \
            -u "${USER_DIR}" -e 10

        bash neural_translation/model_finetune.sh  -l ${langs} \
            -d "${DATA_DIR}/${model}/${seed}" \
            -w "${WORK_DIR}/${model}/${seed}" \
            -p "${PARAMETER_DIR}/default_parameters_${model}.txt" \
            -u "${USER_DIR}" -e 30 # total (train + finetune)

        bash neural_translation/checkpoint_select_best.sh \
            -l ${langs} -r ${langs} \
            -w "${WORK_DIR}/${model}/${seed}" \
            -d "${DATA_DIR}/${model}/${seed}" \
            -u "${USER_DIR}" \
            -n 1 -b 1 -f

        bash neural_translation/bleu_test_save_best.sh \
            -l ${langs} -r ${langs} \
            -w "${WORK_DIR}/${model}/${seed}" \
            -d "${DATA_DIR}/${model}/${seed}" \
            -u "${USER_DIR}" \
            -n 10 -b 10 -f
        done
done