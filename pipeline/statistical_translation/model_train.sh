#!/bin/bash
add_lm=""
while getopts m:w:i:o:l: o
do  case "$o" in
	  m)	MOSES_DIR="$OPTARG";;
    w)  WORK_DIR="$OPTARG";;
    i)  l_in="$OPTARG";;
    o)  l_out="$OPTARG";;
    l)  EXTRA_LANGUAGE_MODEL="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-i source language (input language)]
                                 [-o target language (output language)]
                                 [-w working directory]
                                 [-m path to moses]
                                 [-l supplementary language model path - optional (for fine tuning)]
                                 "
		exit 1;;
	esac
done

# Default LM path
LM_PATH=${WORK_DIR}/${l_in}_${l_out}/lm

if ! [[ ${EXTRA_LANGUAGE_MODEL} == "" ]]; then
    add_lm="-lm 0:3:${EXTRA_LANGUAGE_MODEL}/${l_in}_${l_out}/lm/train_${l_in}_${l_out}_language_model.blm.${l_out}:8";
fi

nohup nice ${MOSES_DIR}/mosesdecoder/scripts/training/train-model.perl \
      -mgiza -external-bin-dir="${MOSES_DIR}/mgiza/mgizapp/bin" \
      -root-dir "${WORK_DIR}/${l_in}_${l_out}/train" -corpus ${LM_PATH}/train_${l_in}_${l_out}.clean -f ${l_in} -e ${l_out} \
      -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
      -lm 0:3:${LM_PATH}/train_${l_in}_${l_out}_language_model.blm.${l_out}:8 \
      ${add_lm}
