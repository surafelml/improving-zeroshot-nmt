#!/bin/bash

EXPDIR=$PWD

# mosesdecoder
if [ ! -d $EXPDIR/mosesdecoder ]; then 
  git clone https://github.com/moses-smt/mosesdecoder.git
fi


# fairseq
if [ ! -d $EXPDIR/fairseq ]; then 
 git clone https://github.com/pytorch/fairseq
 pushd fairseq
 pip install --editable ./
 popd
fi


# subword-nmt 
# git clone https://github.com/rsennrich/subword-nmt.git
pip install subword-nmt
