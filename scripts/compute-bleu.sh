#!/bin/bash

HYP=$1
REF=$2

EXPDIR=$PWD
MOSES=$EXPDIR/mosesdecoder/scripts
MULTIBLUE=$MOSES/generic/multi-bleu.perl

$MULTIBLUE ${REF} < ${HYP} | cut -f 3 -d ' ' | cut -f 1 -d ','

#BLEU=`$MULTIBLUE ${REF} < ${HYP} | cut -f 3 -d ' ' | cut -f 1 -d ','` 
#echo "ROUND: 0 $ROUND - 1 | ZST_DIR: $SRC-$TGT | MULTIBLEU=$BLEU" | tee -a $ZST_MODEL/zst_evaluation.log


