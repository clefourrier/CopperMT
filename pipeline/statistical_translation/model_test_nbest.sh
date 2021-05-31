#!/bin/bash

while getopts n:m:w:d:i:o: o
do  case "$o" in
	m)	MOSES_DIR="$OPTARG";;
    d)  DATA_DIR="$OPTARG";;
    w)  WORK_DIR="$OPTARG";;
    i)  l_in="$OPTARG";;
    o)  l_out="$OPTARG";;
    n)  n_best="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-i source language (input language)]
                                 [-o target language (output language)]
                                 [-w working directory]
                                 [-d data directory]
                                 [-m path to moses]
                                 [-n n_best value]
                                 "
		exit 1;;
	esac
done

echo "nice ${MOSES_DIR}/mosesdecoder/bin/moses \
     -f ${WORK_DIR}/${l_in}_${l_out}/train/model/moses.ini \
     -n-best-list ${WORK_DIR}/out/test_${l_in}_${l_out}_nbest_${n_best}.${l_out} \
     ${n_best} distinct < ${DATA_DIR}/test_${l_in}_${l_out}.${l_in} \
     > ${WORK_DIR}/${l_in}_${l_out}/out/test_${l_in}_${l_out}_translated_${n_best}.${l_out}"

nohup nice ${MOSES_DIR}/mosesdecoder/bin/moses \
     -f ${WORK_DIR}/${l_in}_${l_out}/train/model/moses.ini \
     -n-best-list ${WORK_DIR}/${l_in}_${l_out}/out/test_${l_in}_${l_out}_nbest_${n_best}.${l_out} \
     ${n_best} distinct < ${DATA_DIR}/test_${l_in}_${l_out}.${l_in} \
     > ${WORK_DIR}/${l_in}_${l_out}/out/test_${l_in}_${l_out}_translated_${n_best}.${l_out}
