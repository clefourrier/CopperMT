#!/bin/bash
source $1
export WK_DIR INPUTS_DIR MOSES_DIR DATA_NAME langs_bi
source ../pyenv/bin/activate

DATA_DIR="${INPUTS_DIR}/split_data/shared_${DATA_NAME}"  # Where the data will be saved
WORK_DIR="${WK_DIR}/shared_statistical/"  # Where the models will be saved

# CREATING NEEDED FOLDERS
for lang_pair in $(echo ${langs_bi} | tr "," "\n"); do
    mkdir -p ${WORK_DIR}
    for splits in 0.10 0.20 0.30 0.40 0.50; do
        IFS="-" read l_in l_out <<< "${lang_pair}";
        mkdir -p "${WORK_DIR}/${splits}/${l_in}-${l_out}/lm"
        mkdir -p "${WORK_DIR}/${splits}/${l_out}-${l_in}/lm"
        mkdir -p "${WORK_DIR}/${splits}/${l_in}-${l_out}/out"
        mkdir -p "${WORK_DIR}/${splits}/${l_out}-${l_in}/out"
    done
done

for splits in 0.10 0.20 0.30 0.40 0.50; do
    for lang_pair in $(echo ${langs_bi} | tr "," "\n"); do
        IFS="-" read l_in l_out <<< "${lang_pair}";
        echo "========== PREPROCESS"
        bash statistical_translation/data_preprocess_lm.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${splits}" \
            -d "${DATA_DIR}/${splits}"
        echo "========== TRAIN"

        bash statistical_translation/model_train.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${splits}"

        echo "========== FINETUNE"
        bash statistical_translation/model_finetune.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${splits}" \
            -d "${DATA_DIR}/${splits}"

        bash statistical_translation/model_test_nbest.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${splits}" \
            -d "${DATA_DIR}/${splits}" -n 10

    done
done
