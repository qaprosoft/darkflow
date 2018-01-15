#!/bin/bash

export SCRIPTS_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export MODEL=$1
export IMAGE_HOME=$2
export OUTPUT_TYPE=$3


if [ $# -lt 3 ]; then
    printf 'Usage: %s MODEL_NAME IMAGE_HOME OUTPUT_TYPE [img, json, xml] \n' "$(basename "$0")" >&2
    exit -1
fi

if [ -d $IMAGE_HOME/out ]; then
    echo removing $IMAGE_HOME/out folder.
    rm -rf $IMAGE_HOME/out
fi

echo running recognition command against $IMAGE_HOME
python $SCRIPTS_HOME/recognize.py --darkflow_home $DARKFLOW_HOME --model $MODEL --folder $IMAGE_HOME --output $OUTPUT_TYPE

