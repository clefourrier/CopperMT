#!/bin/bash

while getopts "l:r:d:s:t:v:" o
do  case "$o" in
	l)	LANGS="$OPTARG"; LANGS=$(echo ${LANGS} | tr "," "\n");;
    r)  RAW_DATA_DIR="$OPTARG";;
    d)  DATA_DIR="$OPTARG";;
    s)  SEEDS="$OPTARG"; SEEDS=$(echo ${SEEDS} | tr " " "\n");;
    t)  TRAIN_SPLIT="$OPTARG";;
    v)  VAL_TEST_SPLIT="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-l lang pairs (format l_in-l_out,l_in2-l_out2...)]
                                 [-r original raw data dir (format folder/lin_lout.lin, ...)]
                                 [-d target split data dir (final format will be folder/train_lin_lout.lin... - or fine_tune or test)]
                                 [-s shuffling seeds to use]
                                 [-t train to dev/test split, in % (ex: 70 will mean 70% train, 30% dev/test]
                                 [-v val to test split, in % (ex: 50 will mean 50% dev, 50% test, of the above total dev/test 30%)]"
		exit 1;;
	esac
done

get_seeded_random(){
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

mkdir -p $DATA_DIR
for lang_pair in ${LANGS}; do
    IFS="-" read l_in l_out <<< "${lang_pair}";
    for seed in ${SEEDS}; do
        mkdir "${DATA_DIR}/${seed}"
        # Union and shuffle of data
        paste -d ';' "${RAW_DATA_DIR}/${l_in}_${l_out}.${l_in}" "${RAW_DATA_DIR}/${l_in}_${l_out}.${l_out}" | \
        shuf --random-source=<(get_seeded_random seed) > "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_shuffled"

        # Split
        csplit -f "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_train_vs_rest" \
            "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_shuffled" \
            $(( $(wc -l < "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_shuffled") * TRAIN_SPLIT / 100 + 1))
        csplit -f "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_val_vs_test" \
            "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_train_vs_rest01" \
            $(( $(wc -l < "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_train_vs_rest01") * VAL_TEST_SPLIT / 100 + 1))

        # Split the files back to their respective languages
        awk -v in_path="${DATA_DIR}/${seed}/train.${l_in}-${l_out}.${l_in}" \
            -v out_path="${DATA_DIR}/${seed}/train.${l_in}-${l_out}.${l_out}" \
            'BEGIN { FS=";" } { print $1 > in_path; print $2 > out_path}' \
            < "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_train_vs_rest00"

        awk -v in_path="${DATA_DIR}/${seed}/valid.${l_in}-${l_out}.${l_in}" \
            -v out_path="${DATA_DIR}/${seed}/valid.${l_in}-${l_out}.${l_out}" \
            'BEGIN { FS=";" } { print $1 > in_path; print $2 > out_path}' \
            < "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_val_vs_test00"

        awk -v in_path="${DATA_DIR}/${seed}/test.${l_in}-${l_out}.${l_in}" \
            -v out_path="${DATA_DIR}/${seed}/test.${l_in}-${l_out}.${l_out}" \
            'BEGIN { FS=";" } { print $1 > in_path; print $2 > out_path}' \
            < "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_val_vs_test01"
        rm "${DATA_DIR}/${seed}/preprocess_${l_in}_${l_out}_"*
    done
done