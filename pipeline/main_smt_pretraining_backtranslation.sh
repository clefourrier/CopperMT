#!/bin/bash
source $1
export WK_DIR INPUTS_DIR MOSES_DIR DATA_NAME DATA_MONO_BT_ORIG langs

WORK_DIR="${WK_DIR}/testing_statistical_pretraining_backtrad"  # Where the models will be saved
WORK_DIR_PT="${WORK_DIR}/reference_preds"  # Where the models will be saved
DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Where the data will be saved
ORIGIN_DATA_RAW="${INPUTS_DIR}/raw_data/${DATA_MONO_BT_ORIG}"  # Link to the original data files
DATA_MONO_DIR="${WORK_DIR}/data_mono"  # Where the data will be read by the backtranslation generation models
mkdir -p "${INPUTS_DIR}/raw_data/backtrad/SMT"
RAW_DATA_BACKTRAD="${INPUTS_DIR}/raw_data/backtrad/SMT"  # Where the raw backtranslated data will be saved afterr extraction
SPLIT_DATA_BACKTRAD="${INPUTS_DIR}/split_data/backtrad/SMT"
REFERENCE_MODEL="${WK_DIR}/reference_models/statistical"

# CREATING NEEDED FOLDERS
for lang_pair in $(echo ${langs} | tr "," "\n"); do
    mkdir -p ${WORK_DIR}
    for seed in 0 1 2; do
        IFS="-" read l_in l_out <<< "${lang_pair}";
        mkdir -p "${WORK_DIR}/${seed}/${l_in}_${l_out}/lm"
        mkdir -p "${WORK_DIR}/${seed}/${l_out}_${l_in}/lm"
        mkdir -p "${WORK_DIR}/${seed}/${l_in}_${l_out}/out"
        mkdir -p "${WORK_DIR}/${seed}/${l_out}_${l_in}/out"
        mkdir -p "${WORK_DIR}/${seed}/mono"
    done
done

# Generate backtranslated data
cp -a "${REFERENCE_MODEL}" "${WORK_DIR_PT}"
mkdir -p "${DATA_MONO_DIR}"
for lang_pair in $(echo ${langs} | tr "," "\n"); do
    # 1) We copy the raw source to target data that we will use fully
    IFS="-" read l_in l_out <<< "${lang_pair}";

    cp "${ORIGIN_DATA_RAW}/${l_in}_${l_out}.${l_in}" "${DATA_MONO_DIR}/test_${l_in}_${l_out}.${l_in}"
    cp "${ORIGIN_DATA_RAW}/${l_out}_${l_out}.${l_out}" "${DATA_MONO_DIR}/${l_out}_${l_out}.${l_out}"
done

for seed in 0 1 2; do
    # Predicting backtranslated datasets and extracting backtranslated information
    for lang_pair in $(echo ${langs} | tr "," "\n"); do
        echo "------- ${lang_pair} ${seed}"
        IFS="-" read l_in l_out <<< "${lang_pair}";
        # 2) We use the old reference models to generate new data (very long step)
        bash statistical_translation/model_test_nbest.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR_PT}/${seed}" \
            -d "${DATA_MONO_DIR}" -n 10

        mkdir -p "${RAW_DATA_BACKTRAD}/stat_${lang_pair}/${seed}"

        # 3) We extract said data
        python data/dataraw_extract_from_backtranslated.py \
            "${lang_pair}" "${DATA_MONO_DIR}" \
            "${WORK_DIR_PT}/${seed}" \
            "${RAW_DATA_BACKTRAD}/stat_${lang_pair}/${seed}" 0

        IFS="-" read l_in l_out <<< "${lang_pair}";

        mkdir -p "${SPLIT_DATA_BACKTRAD}/stat_${l_out}-${l_in}/${seed}"
        cp "${RAW_DATA_BACKTRAD}/stat_${lang_pair}/${seed}/${l_out}_${l_in}.${l_in}" \
            "${SPLIT_DATA_BACKTRAD}/stat_${l_out}-${l_in}/${seed}/train_${l_out}_${l_in}.${l_in}" 
        cp "${RAW_DATA_BACKTRAD}/stat_${lang_pair}/${seed}/${l_out}_${l_in}.${l_out}" \
            "${SPLIT_DATA_BACKTRAD}/stat_${l_out}-${l_in}/${seed}/train_${l_out}_${l_in}.${l_out}" \
            
        bash data/data_concatenate.sh \
            -d "${DATA_DIR}/${seed}" \
            -t "${SPLIT_DATA_BACKTRAD}/stat_${l_out}-${l_in}/${seed}"
    done
done

# 4) We train new language models on both new and original data
for seed in 0 1 2; do
    for lang_pair in $(echo ${langs} | tr "," "\n"); do
        IFS="-" read l_in l_out <<< "${lang_pair}";

        bash statistical_translation/data_preprocess_lm.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${seed}" \
            -d "${SPLIT_DATA_BACKTRAD}/stat_${lang_pair}/${seed}"

        bash statistical_translation/model_train.sh \
            -i "${l_in}" -o "${l_out}" \
            -m "${MOSES_DIR}" -w "${WORK_DIR}/${seed}"

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
