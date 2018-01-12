#!/bin/bash

#set of shell commands to build darkflow and sync with cfg/ckpt etc data

export DARKFLOW_HOME=/darkflow/darkflow
export ALICE_HOME=/darkflow/alice

echo DARKFLOW_HOME $DARKFLOW_HOME
echo ALICE_HOME: $ALICE_HOME

cd $DARKFLOW_HOME
# build the Cython extensions in place.
python3 setup.py build_ext --inplace


ln -s -f $ALICE_HOME/data/darkflow/ckpt $DARKFLOW_HOME
ln -s -f $ALICE_HOME/data/darkflow/cfg $DARKFLOW_HOME


echo making symlinks for all labels files
find $ALICE_HOME/data/darkflow -name 'labels-*.txt' -exec ln -vsf "{}" $DARKFLOW_HOME ';'


