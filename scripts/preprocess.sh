#!/bin/bash

set -e

# === UPDATE ACCORDINGLY ===
SRCS=$1 #"it ro"  	# different sources, initial paper includes 'de nl it ro'
TGTS="en"		# fixed since ./data/ted-data/ANY-en dir structure 
#BIDIRECTION=true	# if true, aggregate data both in src<>tgt directions
ADD_LANGID=true 	# if true, adds tgt lang_id on src side examples, set false for single pair settings
BPESIZE=32000


MOSES=$PWD/mosesdecoder/scripts
NORM=$MOSES/tokenizer/normalize-punctuation.perl
TOK=$MOSES/tokenizer/tokenizer.perl
DEES=$MOSES//tokenizer/deescape-special-chars.perl


DATA=$PWD/data/ted-data 		
EXPDIR=$PWD/pretrain-model

PRE_DATA=$EXPDIR/data/pre-data
BPE_MODEL=$EXPDIR/data/bpe-model
BPE_DATA=$EXPDIR/data/bpe-data
BIN_DATA=$EXPDIR/data/bin-data
# ===	===


# preprocess 
if [ ! -d $PRE_DATA ]; then 
mkdir $PRE_DATA 
pushd $PRE_DATA

for SRC in $SRCS; do
  for TGT in $TGTS; do
 
    if [ $SRC != $TGT ]; then # && [ ! -d $PRE_DATA ]; then 
      echo "PREPROCESSING $SRC <> $TGT DATA: $PWD"

      for SET in train dev test; do
	RAW_DATA=$DATA/$SRC-$TGT/${SET}

	# if adding lang_id, data should aggregate in bidirectional SRC<>TGT
	if $ADD_LANGID; then 
          $NORM < ${RAW_DATA}.$SRC | $TOK -l $SRC -q | $DEES | awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag" "$0 }' >> ${SET}.src	#$SRC 
          $NORM < ${RAW_DATA}.$TGT | $TOK -l $TGT -q | $DEES | awk -vtgt_tag="<2${SRC}>" '{ print tgt_tag" "$0 }' >> ${SET}.src	#$SRC 

          $NORM < ${RAW_DATA}.$TGT | $TOK -l $TGT -q | $DEES >> ${SET}.tgt
          $NORM < ${RAW_DATA}.$SRC | $TOK -l $SRC -q | $DEES >> ${SET}.tgt	
	else
          $NORM < ${RAW_DATA}.$SRC | $TOK -l $SRC -q | $DEES >> ${SET}.src
          $NORM < ${RAW_DATA}.$TGT | $TOK -l $TGT -q | $DEES >> ${SET}.tgt
	fi

      done

    fi

  done
done

popd
fi


# learn bpe model with training data
if [ ! -d $BPE_MODEL ]; then  
  mkdir $BPE_MODEL
  echo "LEARNING BPE MODEL: $BPE_MODEL"
  subword-nmt learn-joint-bpe-and-vocab --input $PRE_DATA/train.src $PRE_DATA/train.tgt \
					-s $BPESIZE -o $BPE_MODEL/code.${BPESIZE}.bpe \
					--write-vocabulary $BPE_MODEL/train.src.vocab $BPE_MODEL/train.tgt.vocab 
fi
wait $!



# apply sub-word segmentation
if [ ! -d $BPE_DATA ]; then
 mkdir $BPE_DATA

 for SET in train dev test; do
  subword-nmt apply-bpe -c $BPE_MODEL/code.${BPESIZE}.bpe < $PRE_DATA/${SET}.src > $BPE_DATA/${SET}.src 
  subword-nmt apply-bpe -c $BPE_MODEL/code.${BPESIZE}.bpe < $PRE_DATA/${SET}.tgt > $BPE_DATA/${SET}.tgt
 done
fi



# binarize train/valid/test
if [ ! -d $BIN_DATA ]; then
  mkdir $BIN_DATA
  python $FAIRSEQ/preprocess.py -s src -t tgt \
				--destdir $BIN_DATA \
				--trainpref $BPE_DATA/train \
				--validpref $BPE_DATA/dev \
				--testpref $BPE_DATA/test \
				--joined-dictionary \
				--workers 32 
fi
