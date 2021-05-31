#!/bin/bash

while getopts l:o:d:m: o
do  case "$o" in
	l)	LANGS="$OPTARG"; LANGS=$(echo ${LANGS} | tr "," "\n");;
    o)  ORIG_DATA="$OPTARG";;
    d)  DATA_DIR="$OPTARG";;
    m)  MODE="$OPTARG";;  # all-to-all, dec-to-dec, enc-to-enc
    [?])	print >&2 "Usage: $0 [-l lang pairs (format l_in-l_out,l_in2-l_out2...)]
                                 [-o original data dir (format folder/lang_lang.lang, ...)]
                                 [-d target data dir (final format will be folder/lin_lout.lin)]
                                 [-m mode of monolingual data assembly
                                    (dec-to-dec takes target lang data as source and target
                                    enc-to-enc takes source lang data as source and target
                                    all-to-all concatenates all lang data as source and target)]"
		exit 1;;
	esac
done

UNIQ_LANGS=$(for l in $(echo ${LANGS} | tr "-" "\n");do echo $l; done | sort | uniq)

mkdir $DATA_DIR
if [[ ${MODE} == "dec-to-dec" ]]; then
    # We use the decoder data as encoder data
    for lang_pair in ${LANGS}; do
        IFS="-" read l_in l_out <<< "${lang_pair}";
        cp "${ORIG_DATA}/${l_out}_${l_out}.${l_out}" "${DATA_DIR}/${l_in}_${l_out}.${l_in}"
        cp "${ORIG_DATA}/${l_out}_${l_out}.${l_out}" "${DATA_DIR}/${l_in}_${l_out}.${l_out}"
    done

elif [[ ${MODE} == "enc-to-enc" ]]; then
    for lang_pair in ${LANGS}; do
        IFS="-" read l_in l_out <<< "${lang_pair}";
        cp "${ORIG_DATA}/${l_in}_${l_in}.${l_in}" "${DATA_DIR}/${l_in}_${l_out}.${l_in}"
        cp "${ORIG_DATA}/${l_in}_${l_in}.${l_in}" "${DATA_DIR}/${l_in}_${l_out}.${l_out}"
    done

elif [[ ${MODE} == "all-to-all" ]]; then
    for lang in ${UNIQ_LANGS}; do
        cat "${ORIG_DATA}/${lang}_${lang}.${lang}" >> "${ORIG_DATA}/full"
    done
    for lang_pair in ${LANGS}; do
        IFS="-" read l_in l_out <<< "${lang_pair}";
            cat "${ORIG_DATA}/full" > "${DATA_DIR}/${l_in}_${l_out}.${l_in}"
            cat "${ORIG_DATA}/full" > "${DATA_DIR}/${l_in}_${l_out}.${l_out}"
    done
    rm "${ORIG_DATA}/full"
fi