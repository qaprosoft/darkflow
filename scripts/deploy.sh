#!/bin/bash

#set of shell commands to build darkflow and sync with cfg/ckpt etc data

export DARKFLOW_HOME=/qps-ai/darkflow
export AI_DATA=/qps-ai/data

echo DARKFLOW_HOME $DARKFLOW_HOME
echo AI_DATA: $AI_DATA

cd $DARKFLOW_HOME
# build the Cython extensions in place.
python3 setup.py build_ext --inplace


ln -s -f $AI_DATA/darkflow/ckpt $DARKFLOW_HOME
ln -s -f $AI_DATA/darkflow/cfg $DARKFLOW_HOME


echo making symlinks for all labels files
find $AI_DATA/darkflow -name 'labels-*.txt' -exec ln -vsf "{}" $DARKFLOW_HOME ';'


