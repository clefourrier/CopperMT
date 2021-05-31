#!/bin/bash
source $1
export WK_DIR INPUTS_DIR DATA_NAME langs

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_DIR="${DIR}/neural_translation/multilingual_rnns"  # Link to "multilingual_rnns"

# ------ PARAMETERS
# INPUTS
ORIGIN_DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Link to the original data files
PARAMETER_DIR="${INPUTS_DIR}/parameters/testing"  # Contains the parameter files
# WORKING
DATA_DIR="${WK_DIR}/testing_parameters_bilingual/data"  # Where the data will be saved
WORK_DIR="${WK_DIR}/testing_parameters_bilingual"  # Where the models will be saved

# ------- PREPROCESSING
for seed in 0 1 2; do
    bash ${DIR}/neural_translation/data_preprocess.sh -l ${langs} \
        -o "${ORIGIN_DATA_DIR}/${seed}"\
        -d "${DATA_DIR}/${seed}"
done

# ------- TRAINING RNN AND TRANSFORMER
for lang_pair in $(echo ${langs} | tr "," "\n"); do
    mkdir -p "${PARAMETER_DIR}_${lang_pair}"
    for cur_exp in "emb_vs_hid" "batch_vs_lr" "nb_layers" "compare_attentions" "compare_heads"; do
        python ${DIR}/neural_translation/parameters_create_files.py ${cur_exp} "${PARAMETER_DIR}_${lang_pair}"
        for cur_seed in 0 1 2; do
            for parameter_comb in $(ls "${PARAMETER_DIR}_${lang_pair}/${cur_exp}"); do
                echo ${parameter_comb}
                exp_path="${cur_exp}/${parameter_comb}"
                bash ${DIR}/neural_translation/model_train.sh  -l ${lang_pair} \
                    -d "${DATA_DIR}/${cur_seed}" \
                    -w "${WORK_DIR}_${lang_pair}/${exp_path}/${cur_seed}" \
                    -p "${PARAMETER_DIR}_${lang_pair}/${cur_exp}/${parameter_comb}" \
                    -u "${USER_DIR}" -e 20

                bash ${DIR}/neural_translation/checkpoint_select_best.sh \
                    -l ${lang_pair} -r ${lang_pair} \
                    -w "${WORK_DIR}_${lang_pair}/${exp_path}/${cur_seed}" \
                    -d "${DATA_DIR}/${cur_seed}" \
                    -u "${USER_DIR}" \
                    -n 1 -b 1

                bash ${DIR}/neural_translation/bleu_test_save_best.sh \
                    -l ${lang_pair} -r ${lang_pair} \
                    -w "${WORK_DIR}_${lang_pair}/${exp_path}/${cur_seed}" \
                    -d "${DATA_DIR}/${cur_seed}" \
                    -u "${USER_DIR}" \
                    -n 1 -b 1

                bash ${DIR}/neural_translation/bleu_test_print.sh \
                    -l ${lang_pair} -w "${WORK_DIR}_${lang_pair}/${exp_path}/${cur_seed}" \
                    >> "${WORK_DIR}_${lang_pair}/${exp_path}/${cur_seed}/bleu/best_bleu_synthesis.csv"
            done
        done
        python ${DIR}/neural_translation/parameters_save_best.py ${cur_exp} \
               "${PARAMETER_DIR}_${lang_pair}" "${WORK_DIR}_${lang_pair}/${cur_exp}" "0 1 2" ${lang_pair}
    done
done