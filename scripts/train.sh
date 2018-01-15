#/bin/bash

export SCRIPTS_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )


export MODEL=$1

if [ $# -lt 1 ]; then
    printf 'Usage: %s MODEL_NAME\n' "$(basename "$0")" >&2
    exit -1
fi


export DATA_HOME=$ALICE_HOME/data/$MODEL
echo DATA_HOME: $DATA_HOME

#make temporary folder with symlinks onto the train set data (img and ann)
export TMP_DATA=/tmp/train/$MODEL

#clean existing $TMP_DATA folder if any
if [ -d $TMP_DATA ]; then
        echo Removing $TMP_DATA...
        rm -rf $TMP_DATA
	echo creating clean $TMP_DATA folder
fi


mkdir -p $TMP_DATA
#recursively create symlinks from $DATA_HOME to $TMP_DATA
mkdir $TMP_DATA/img
mkdir $TMP_DATA/ann

find $DATA_HOME -name '*.png' -exec ln -vs "{}" $TMP_DATA/img/ ';'
find $DATA_HOME -name '*.xml' -exec ln -vs "{}" $TMP_DATA/ann/ ';'

nohup $DARKFLOW_HOME/flow --train --labels $DARKFLOW_HOME/labels-$MODEL.txt --annotation $TMP_DATA/ann --dataset $TMP_DATA/img --model $DARKFLOW_HOME/cfg/$MODEL.cfg  \
	--load -1 --trainer adam --gpu 0.9 --lr 1e-5 --keep 500 --backup $DARKFLOW_HOME/ckpt/$MODEL/ --batch 8 --save 2900 --epoch 3000 > ../logs/train_$MODEL.log &
