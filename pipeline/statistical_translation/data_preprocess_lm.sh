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

echo "${MOSES_DIR} ${DATA_DIR} ${WORK_DIR} ${l_in}"
if [[ ${WORK_DIR} == "" ]]; then
    exit 1;
fi

mkdir -p ${WORK_DIR}/${l_in}_${l_out}/lm
LM_PATH=${WORK_DIR}/${l_in}_${l_out}/lm

${MOSES_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl \
    ${DATA_DIR}/train_${l_in}_${l_out} ${l_in} ${l_out} \
    ${LM_PATH}/train_${l_in}_${l_out}.clean 1 80
${MOSES_DIR}/mosesdecoder/bin/lmplz -o 3 --discount_fallback \
    <${DATA_DIR}/train_${l_in}_${l_out}.${l_out} > \
    ${LM_PATH}/train_${l_in}_${l_out}_language_model.arpa.${l_out}
${MOSES_DIR}/mosesdecoder/bin/build_binary \
    ${LM_PATH}/train_${l_in}_${l_out}_language_model.arpa.${l_out} \
    ${LM_PATH}/train_${l_in}_${l_out}_language_model.blm.${l_out}
echo "---- Clean and build data done"