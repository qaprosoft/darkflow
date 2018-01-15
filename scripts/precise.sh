#!/bin/bash

export SCRIPTS_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Start recognition for appropriate item(s) and calculate precision accuracy based on ideal metadata

export MODEL=$1

if [ $# -lt 1 ]; then
    printf 'Usage: %s MODEL_NAME\n' "$(basename "$0")" >&2
    exit -1
fi


export DATA_HOME=$AI_DATA/$MODEL
echo DATA_HOME: $DATA_HOME

#make temporary folder with symlinks onto the train set data (img and ann)
export TMP_DATA=/tmp/precise/$MODEL

#clean existing $TMP_DATA folder if any
if [ -d $TMP_DATA ]; then
        echo Removing $TMP_DATA...
        rm -rf $TMP_DATA
        echo creating clean $TMP_DATA folder
fi


mkdir -p $TMP_DATA
echo recursively create symlinks from $DATA_HOME to $TMP_DATA
mkdir $TMP_DATA/img
mkdir $TMP_DATA/ann

find $DATA_HOME -name '*.png' -exec ln -vs "{}" $TMP_DATA/img/ ';'
find $DATA_HOME -name '*.xml' -exec ln -vs "{}" $TMP_DATA/ann/ ';'

echo $TMP_DATA/img


export IMAGE_HOME=$TMP_DATA/img
export ANNOTATION_HOME=$TMP_DATA/ann

echo running recognition command against $IMAGE_HOME
python $SCRIPTS_HOME/recognize.py --darkflow_home $DARKFLOW_HOME --model $MODEL --folder $IMAGE_HOME --output json --labels $DARKFLOW_HOME/labels-$MODEL.txt

echo running precision accuracy calculation script
python $SCRIPTS_HOME/precise.py --truth $ANNOTATION_HOME --predicted $IMAGE_HOME/out --labels $DARKFLOW_HOME/labels-$MODEL.txt
