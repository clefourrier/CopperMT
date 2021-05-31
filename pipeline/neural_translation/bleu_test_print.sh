#!/bin/bash

while getopts w:l: o
do  case "$o" in
	l)	LANGS="$OPTARG"; LANGS=$(echo ${LANGS} | tr "," "\n");;
    w)  WORK_DIR="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-l lang pairs (format l_in-l_out,l_in2-l_out2...)]
                                 [-w working directory]"
		exit 1;;
	esac
done

for lang_pair in ${LANGS}; do
    IFS="-" read l_in l_out <<< "${lang_pair}";
    bleu=$(tail -n1 "${WORK_DIR}/bleu/bleu_checkpoint_best_${l_in}_${l_out}.${l_out}" | \
           awk -F 'BLEU = ' '{print $2"\n"}'  | \
           awk -F '(' '{printf $1"\n"}' | awk 'NF > 0')
    echo "$l_in $l_out $bleu";
done