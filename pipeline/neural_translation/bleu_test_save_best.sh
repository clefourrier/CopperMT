#!/bin/bash

DATA_BIN="data-bin";
CP="checkpoints";

while getopts w:l:fn:b:r:d:u: o
do  case "$o" in
	l)	LANGS="$OPTARG"; LANGS=$(echo ${LANGS} | tr "," "\n");;
    r)  REF_MODEL_LANGS="$OPTARG";;
    u)  USR_DIR="$OPTARG";;
    w)  WORK_DIR="$OPTARG";;
    d)  DATA_DIR="$OPTARG";;
    f)  DATA_BIN="data-bin-finetune"; CP="checkpoints-finetune";;
    n)  nbest="$OPTARG";;
    b)  beam="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-l lang pairs of interest (format l_in-l_out,l_in2-l_out2...)]
                                 [-r all model lang pairs (same format)]
                                 [-u user directory - path to multilingual_rnns]
                                 [-w working directory]
                                 [-d data directory]
                                 [-f (we are looking at fine-tuned data/models)]
                                 [-n nbest dimension]
                                 [-b beam dimension]
                                 "
		exit 1;;
	esac
done

if [[ ${WORK_DIR} == "" ]]; then
    exit 1;
fi

mkdir -p ${WORK_DIR}/bleu

# To get all checkpoints
#for f in $(ls "$WORKSPACE/checkpoints/$EXP_NAME"); do
for lang_pair in ${LANGS}; do
    IFS="-" read l_in l_out <<< "${lang_pair}";
    fairseq-generate ${DATA_DIR}/${DATA_BIN}/ \
    --user-dir ${USR_DIR} --path "${WORK_DIR}/${CP}/checkpoint_best.pt" \
    --batch-size 1000 --beam ${beam} --nbest ${nbest} \
    --task "multilingual_translation" --lang-pairs ${REF_MODEL_LANGS} \
    --scoring sacrebleu -s ${l_in} -t ${l_out} \
    > "${WORK_DIR}/bleu/bleu_checkpoint_best_${l_in}-${l_out}.${l_out}"
    echo "$l_in $l_out DONE"
done
