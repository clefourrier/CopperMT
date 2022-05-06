#!/bin/bash
source $1
export WK_DIR INPUTS_DIR DATA_NAME langs_bi langs langs_shared
source ../pyenv/bin/activate


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SPLITDATA_DIR="${INPUTS_DIR}/split_data/shared_${DATA_NAME}"
DATA_DIR="${INPUTS_DIR}/data/shared_${DATA_NAME}"
mkdir -p ${DATA_DIR}


# Remove le and ld parameters to not share embeddings in encoders/decoders
for split in 0.10 0.20 0.30 0.40 0.50; do
	python3 ${DIR}/data/dataraw_segmentation.py \
	-d=${SPLITDATA_DIR}/${split} \
	-lp=${langs_bi} \
	-l=${langs} \
	-le=${langs_shared} \
	-ld=${langs_shared} \
	-t="char" -v=200 \
	2>&1 | tee log.log

	# Preprocess according to fairseq
	python3 ${DIR}/neural_translation/data_preprocess.py \
		-lp=${langs_bi} \
		-l=${langs} \
		-le=${langs_shared} \
		-ld=${langs_shared} \
	    -o="${SPLITDATA_DIR}/${split}" \
	    -r="${SPLITDATA_DIR}/${split}" \
	    -d="${DATA_DIR}/${split}"
done