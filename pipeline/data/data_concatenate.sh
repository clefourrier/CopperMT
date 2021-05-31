#!/bin/bash

# Both data dirs must contain data already split correctly
while getopts d:t: o
do  case "$o" in
    d)  DATA_DIR="$OPTARG";;
    t)  TARGET_DATA_DIR="$OPTARG";;
    [?])	print >&2 "Usage: $0 [-d data to add] [-t data to add to]"
		exit 1;;
	esac
done

echo "add ${DATA_DIR} to ${TARGET_DATA_DIR}"
for f in $(ls ${TARGET_DATA_DIR}); do
    cat "${DATA_DIR}/${f}" >> "${TARGET_DATA_DIR}/${f}"
done
