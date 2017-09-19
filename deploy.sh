#!/bin/bash

#set of shell commands to build darkflow and sync with cfg/ckpt etc data

if [ $# -lt 1 ]; then
    printf 'Usage: %s ALICE_HOME \n' "$(basename "$0")" >&2
    exit -1
fi


# set DARKFLOW_HOME based on location of deploy.sh to be able to execute it from any directory
export DARKFLOW_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
echo DARKFLOW_HOME $DARKFLOW_HOME

export ALICE_HOME=$1
echo ALICE_HOME: $ALICE_HOME

# build the Cython extensions in place.
python3 setup.py build_ext --inplace


ln -s -f $ALICE_HOME/data/darkflow/ckpt $DARKFLOW_HOME
ln -s -f $ALICE_HOME/data/darkflow/cfg $DARKFLOW_HOME


echo making symlinks for all labels files
find $ALICE_HOME/data/darkflow -name 'labels-*.txt' -exec ln -vsf "{}" $DARKFLOW_HOME ';'


