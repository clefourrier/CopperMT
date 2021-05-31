#!/bin/bash

while getopts m:w:d:i:o: o
do  case "$o" in
	m)	MOSES_DIR="$OPTARG";;
    d)  DATA_DIR="$OPTARG";;
    w)  WORK_DIR="$OPTARG";;
    i)  l_in="$OPTARG";;
    o)  l_out="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-i source language (input language)]
                                 [-o target language (output language)]
                                 [-w working directory]
                                 [-d data directory]
                                 [-m path to moses]
                                 "
		exit 1;;
	esac
done


cd "${WORK_DIR}/${l_in}_${l_out}/train"

nohup nice ${MOSES_DIR}/mosesdecoder/scripts/training/mert-moses.pl \
    ${DATA_DIR}/fine_tune_${l_in}_${l_out}.${l_in} ${DATA_DIR}/fine_tune_${l_in}_${l_out}.${l_out} \
    ${MOSES_DIR}/mosesdecoder/bin/moses ${WORK_DIR}/${l_in}_${l_out}/train/model/moses.ini \
    --mertdir ${MOSES_DIR}/mosesdecoder/bin/ \
	--mertargs="--sctype BLEU" --return-best-dev