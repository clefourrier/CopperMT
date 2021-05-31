#!/bin/bash

while getopts w:l:u:p:d:e: o
do  case "$o" in
	l)	LANGS="$OPTARG";;
    w)  WORK_DIR="$OPTARG";;
    d)  DATA_DIR="$OPTARG";;
    u)  USR_DIR="$OPTARG";;
    p)  SOURCE_FILE="$OPTARG";;
    e)  epoch="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-l lang pairs of interest (format l_in-l_out,l_in2-l_out2...)]
                                 [-u user directory - path to multilingual_rnns]
                                 [-w working directory]
                                 [-d data directory]
                                 [-p parameter_file for the model]
                                 [-e number of training epochs]
                                 "
		exit 1;;
	esac
done

if [[ ${WORK_DIR} == "" ]]; then
    echo "Work dir empty, exiting"
    exit 1;
fi

source ${SOURCE_FILE}
export model_type attention \
    enc_layer enc_emb_dim enc_hid_dim enc_heads \
    dec_layer dec_emb_dim dec_hid_dim dec_heads \
    batch_size dropout learning_rate \
    share_encoders share_decoders

if [[ ${share_encoders} = true ]]; then share_encoders="--share-encoders"; else share_encoders=""; fi
if [[ ${share_decoders} = true ]]; then share_decoders="--share-decoders"; else share_decoders=""; fi

if [[ ${model_type} == "transformer" ]];
then
   encoder_params="--encoder-ffn-embed-dim ${enc_hid_dim} --encoder-attention-heads ${enc_heads}";
   decoder_params="--decoder-ffn-embed-dim ${dec_hid_dim} --decoder-attention-heads ${enc_heads}";
   attention="";
else
   encoder_params="--encoder-hidden-size ${enc_hid_dim}";
   decoder_params="--decoder-hidden-size ${dec_hid_dim}";
   attention="--attention-type ${attention}";
fi

echo "--lr ${learning_rate} --batch-size ${batch_size} --dropout ${dropout} --max-epoch ${epoch} \
  --save-dir ${WORK_DIR}/checkpoints/ --optimizer adam --scoring sacrebleu \
  --user-dir ${USR_DIR} --arch multilingual_${model_type} \
  --task multilingual_translation --lang-pairs ${LANGS} \
  --encoder-layers ${enc_layer} --encoder-embed-dim ${enc_emb_dim} ${encoder_params} \
  --decoder-layers ${dec_layer} --decoder-embed-dim ${dec_emb_dim} ${decoder_params} \
  ${attention} --share-encoders ${share_encoders} --share-decoders ${share_decoders}"

CUDA_VISIBLE_DEVICES=0 fairseq-train "${DATA_DIR}/data-bin/" \
  --lr ${learning_rate} --batch-size ${batch_size} --dropout ${dropout} --max-epoch ${epoch} \
  --save-dir "${WORK_DIR}/checkpoints/" --optimizer adam --scoring sacrebleu \
  --user-dir ${USR_DIR} --arch multilingual_${model_type} \
  --task multilingual_translation --lang-pairs ${LANGS} \
  --encoder-layers ${enc_layer} --encoder-embed-dim ${enc_emb_dim} ${encoder_params} \
  --decoder-layers ${dec_layer} --decoder-embed-dim ${dec_emb_dim} ${decoder_params} \
  ${attention} ${share_encoders} ${share_decoders}