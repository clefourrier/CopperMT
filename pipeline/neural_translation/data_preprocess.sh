#!/bin/bash

DATA_BIN="data-bin"
trainpref="train_"
validpref="fine_tune_"
testpref="test_"
joined_dicts_in=false
joined_dicts_out=false
in_dict=""
out_dict=""
only_source=""

while getopts o:l:d:mfrjh o
do  case "$o" in
	l)	LANGS="$OPTARG"; LANGS=$(echo ${LANGS} | tr "," "\n");;
    d)  DATA_DIR="$OPTARG";;
    o)  ORIG_DATA_DIR="$OPTARG";;
    # Monolingual
    m)  only_source="--only-source";;
    # Fine tuning
    f)  DATA_BIN="data-bin-finetune";;
    r)  trainpref=""; validpref=""; testpref="";;
    j)  joined_dicts_in=true;;  # When sharing encoders
    h)  joined_dicts_out=true;;  # When sharing decoders
    [?])	print >&2 "Usage: $0 [-l lang pairs (format l_in-l_out,l_in2-l_out2...)]
                                 [-d save data directory]
                                 [-o original data directory]
                                 [-m (the data is monolingual and you want only the source to be preprocessed)]
                                 [-f (the data is being preprocessed for fine-tuning (and will be saved in data-bin-finetune))]
                                 [-r (the data is raw (not split))]
                                 [-j (the encoders of the model will be shared)]
                                 [-h (the decoders of the model will be shared)]
                                 "
		exit 1;;
	esac
done

mkdir -p "${DATA_DIR}/${DATA_BIN}"

# With previous data - We copy the data-bin original dict (must be common if it is a shared enc/dec)
if [[ ${DATA_BIN} == "data-bin-finetune" ]]; then
    if [ ${joined_dicts_in} = true ] ; then
        cp "${DATA_DIR}/data-bin/common_dict_in.vocab" "${DATA_DIR}/${DATA_BIN}/common_dict_in.vocab"
    fi
    if [ ${joined_dicts_out} = true ] ; then
        cp "${DATA_DIR}/data-bin/common_dict_out.vocab" "${DATA_DIR}/${DATA_BIN}/common_dict_out.vocab"
    fi
    UNIQ_LANGS=$(for l in $(echo ${LANGS} | tr "-" "\n"); do echo $l; done | sort | uniq)
    for U_LANG in ${UNIQ_LANGS}; do
        if [ -f "${DATA_DIR}/data-bin/dict.${U_LANG}.txt" ]; then
            echo "COPYING DATA DICT cp ${DATA_DIR}/data-bin/dict.${U_LANG}.txt ${DATA_DIR}/${DATA_BIN}/dict.${U_LANG}.txt"
            cp "${DATA_DIR}/data-bin/dict.${U_LANG}.txt" \
               "${DATA_DIR}/${DATA_BIN}/dict.${U_LANG}.txt"
        fi
    done
elif [ ${joined_dicts_in} = true ] || [ ${joined_dicts_out} = true ] ; then # Without
    TRAIN_FILES=$(for lang_pair in ${LANGS}; do IFS="-" read l_in l_out <<< "${lang_pair}";
            echo ${ORIG_DATA_DIR}/train_${l_in}_${l_out}.${l_in}; echo ${ORIG_DATA_DIR}/train_${l_in}_${l_out}.${l_out};
            done | tr "\n" ",")
    python neural_translation/spm_train.py \
        --input=${TRAIN_FILES} \
        --model_prefix="${DATA_DIR}/${DATA_BIN}/common_dict.bpe" \
        --vocab_size=83 \
        --character_coverage=1.0 \
        --model_type=bpe

    # Comes from https://github.com/pytorch/fairseq/issues/859
    if [ ${joined_dicts_in} = true ] ; then
        tail -n +4 "${DATA_DIR}/${DATA_BIN}/common_dict.bpe.vocab" | cut -f1 | sed 's/$/ 100/g' > "${DATA_DIR}/${DATA_BIN}/common_dict_in.vocab"
    fi
    if [ ${joined_dicts_out} = true ] ; then
        tail -n +4 "${DATA_DIR}/${DATA_BIN}/common_dict.bpe.vocab" | cut -f1 | sed 's/$/ 100/g' > "${DATA_DIR}/${DATA_BIN}/common_dict_out.vocab"
    fi
fi


for lang_pair in ${LANGS}; do
    IFS="-" read l_in l_out <<< "${lang_pair}";
    # If we want a joined dict, we create it
    if  [ ${joined_dicts_in} = true ]; then
            in_dict="--srcdict ${DATA_DIR}/${DATA_BIN}/common_dict_in.vocab";
    elif [ -f "${DATA_DIR}/${DATA_BIN}/dict.${l_in}.txt" ]; then
            in_dict="--srcdict ${DATA_DIR}/${DATA_BIN}/dict.${l_in}.txt"
    fi

    if [ ${joined_dicts_out} = true ] ; then
        out_dict="--tgtdict ${DATA_DIR}/${DATA_BIN}/common_dict_out.vocab"
        # If there is already a dict for the relevant target language, we use it
    elif [ -f "${DATA_DIR}/${DATA_BIN}/dict.${l_out}.txt" ]; then
        out_dict="--tgtdict ${DATA_DIR}/${DATA_BIN}/dict.${l_out}.txt"
    fi

    fairseq-preprocess --source-lang ${l_in} --target-lang ${l_out} ${only_source} \
      --trainpref ${ORIG_DATA_DIR}/${trainpref}${l_in}_${l_out} \
      --validpref ${ORIG_DATA_DIR}/${validpref}${l_in}_${l_out} \
      --testpref  ${ORIG_DATA_DIR}/${testpref}${l_in}_${l_out} \
      --destdir ${DATA_DIR}/${DATA_BIN} --task multilingual_translation \
      ${in_dict} ${out_dict}
done