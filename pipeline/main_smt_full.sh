#!/bin/bash
source $1
export WK_DIR INPUTS_DIR MOSES_DIR DATA_NAME langs

DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Where the data will be saved
WORK_DIR="${WK_DIR}/reference_models/statistical"  # Where the models will be saved

# CREATING NEEDED FOLDERS
for lang_pair in $(echo ${langs} | tr "," "\n"); do
    mkdir -p ${WORK_DIR}
    for seed in 0 1 2; do
        IFS="-" read l_in l_out <<< "${lang_pair}";
        mkdir -p "${WORK_DIR}/${seed}/${l_in}_${l_out}/lm"
        mkdir -p "${WORK_DIR}/${seed}/${l_out}_${l_in}/lm"
        mkdir -p "${WORK_DIR}/${seed}/${l_in}_${l_out}/out"
        mkdir -p "${WORK_DIR}/${seed}/${l_out}_${l_in}/out"
    done
done

for seed in 0 1 2; do
    for lang_pair in $(echo ${langs} | tr "," "\n"); do
        IFS="-" read l_in l_out <<< "${lang_pair}";
        echo "========== PREPROCESS"
        bash statistical_translation/data_preprocess_lm.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${seed}" \
            -d "${DATA_DIR}/${seed}"
        echo "========== TRAIN"

        bash statistical_translation/model_train.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${seed}"

        echo "========== FINETUNE"
        bash statistical_translation/model_finetune.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${seed}" \
            -d "${DATA_DIR}/${seed}"

        bash statistical_translation/model_test_nbest.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${seed}" \
            -d "${DATA_DIR}/${seed}" -n 10

    done
done
