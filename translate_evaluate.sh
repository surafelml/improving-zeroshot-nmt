#!/bin/bash

#
# takes raw input, translate ed evaluate bleu for SRC-TGT TEST/DEV, translates in SRC-PIVOT-TGT if --pivot is specified 
#

set -e 

EXPDIR=$PWD
DATA_BIN=$1
INPUT=$2 	# raw test file without posfix (.SRC/.TGT)
MODEL=$3	
DEVICE=$4

SRC=$5
PIVOT=$6
TGT=$7

DATA=$EXPDIR/data/ted-data
PRETRAINING=$EXPDIR/pretrain-baseline
BPE_MODEL=$PRETRAINING/data/bpe-model/code.32000.bpe

EVALDATA=$DATA/$SRC-$PIVOT-$TGT
TEST_SRC=$EVALDATA/test.$SRC
TEST_TGT=$EVALDATA/test.$TGT

EVALDIR=$EXPDIR/evaluation
mkdir -p $EVALDIR
LOG_EVAL=$EVALDIR/log.eval


TEST_SRC=${INPUT}.$SRC
TEST_TGT=${INPUT}.$TGT

PARA_TEST_SRC=$EVALDIR/test.$SRC
PARA_TEST_SRC_TAG=$EVALDIR/test.tag.$SRC
PARA_TEST_TGT=$EVALDIR/test.$TGT
PARA_TEST_TGT_TAG=$EVALDIR/test.tag.$TGT

# scrips
MOSES=$EXPDIR/mosesdecoder/scripts
NORM=$MOSES/tokenizer/normalize-punctuation.perl
TOK=$MOSES/tokenizer/tokenizer.perl
DEES=$MOSES//tokenizer/deescape-special-chars.perl
MULTIBLUE=$MOSES/generic/multi-bleu.perl
INFERENCE=$EXPDIR/inference.sh
CBLEU=$EXPDIR/scripts/compute-bleu.sh





# pivot x-p-y inference
if [[ -n "$PIVOT" -a $PIVOT = 'en' ]]; then # 'en' is the pivot language
  echo "INFERENCE WITH PIVOTING: $SRC - $PIVOT - $TGT"

  TEST_PIVOT=${INPUT}.$PIVOT
  PARA_TEST_PIVOT=$EVALDIR/test.$PIVOT
  $NORM < $TEST_SRC | $TOK -l $SRC -q | $DEES \
	| awk -vtgt_tag="<2${PIVOT}>" '{ print tgt_tag" "$0 }' \
	| subword-nmt apply-bpe -c $BPE_MODEL > $PARA_TEST_SRC_TAG
  $NORM < $TEST_TGT | $TOK -l $TGT -q | $DEES  > $PARA_TEST_TGT
  $NORM < $TEST_PIVOT | $TOK -l $PIVOT -q | $DEES  > $PARA_TEST_PIVOT # (optional)


  # inference src-pivot
  bash $INFERENCE $BIN_DATA_ZST $PARA_TEST_SRC_TAG $ZST_CKPT $GPUS
  mv ${PARA_TEST_SRC_TAG}.op ${PARA_TEST_SRC_TAG}.$SRC-$PIVOT

  BLEU=$(bash $CBLEU ${PARA_TEST_SRC_TAG}.$SRC-$PIVOT ${PARA_TEST_PIVOT}) 
  echo "MODEL: $MODEL | DIR: $SRC-$PIVOT | MULTIBLEU=$BLEU" | tee -a $LOG_EVAL

  
  # preprocess 
  PARA_TEST_PIVOT_TAG=$EVALDIR/test.tag.$SRC-$PIVOT
  $NORM < ${PARA_TEST_SRC_TAG}.$SRC-$PIVOT | $TOK -l $PIVOT -q | $DEES \
	| awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag" "$0 }' \
	| subword-nmt apply-bpe -c $BPE_MODEL > $PARA_TEST_PIVOT_TAG


  # inference pivot-tgt 
  bash $INFERENCE $BIN_DATA_ZST $PARA_TEST_PIVOT_TAG $ZST_CKPT $GPUS
  mv ${PARA_TEST_PIVOT_TAG}.op ${PARA_TEST_PIVOT_TAG}.$SRC-$PIVOT-$TGT

  BLEU=$(bash $CBLEU ${PARA_TEST_SRC_TAG}.$SRC-$PIVOT-$TGT ${PARA_TEST_TGT}) 
  echo "MODEL: $MODEL | DIR: $SRC-$PIVOT-$TGT | MULTIBLEU=$BLEU" | tee -a $LOG_EVAL



else # do src-tgt inference 

  echo "INFERENCE FOR $SRC - $TGT ..."
  $NORM < $TEST_SRC | $TOK -l $SRC -q | $DEES \
	| awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag" "$0 }' \
	| subword-nmt apply-bpe -c $BPE_MODEL > $PARA_TEST_SRC_TAG
  $NORM < $TEST_TGT | $TOK -l $TGT -q | $DEES  > $PARA_TEST_TGT

  # inference
  bash $INFERENCE $BIN_DATA_ZST $PARA_TEST_SRC_TAG $ZST_CKPT $GPUS
  mv ${PARA_TEST_SRC_TAG}.op ${PARA_TEST_SRC_TAG}.$SRC-$TGT

  # evaluate 
  BLEU=$(bash $CBLEU ${PARA_TEST_SRC_TAG}.$SRC-$TGT ${PARA_TEST_TGT}) 
  echo "MODEL: $MODEL | ZST_DIR: $SRC-$TGT | MULTIBLEU=$BLEU" | tee -a $LOG_EVAL
fi
