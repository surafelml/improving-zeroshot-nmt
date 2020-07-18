#!/bin/bash

SRC='src'
TGT='tgt' 	# despite different src ed tgt, we keep for getting vocab extentions
DATA_BIN=$1	# dir containing the vocab	
INPUT=$2 	# preprocessed source file
MODEL=$3
DEVICE=$4


CUDA_VISIBLE_DEVICES=$DEVICE fairseq-interactive $DATA_BIN -s $SRC -t $TGT \
		--batch-size 128 \
		--buffer-size 4096 \
		--beam 5 \
		--input $INPUT \
		--path $MODEL \
		--remove-bpe \
		--no-progress-bar \
		| grep "^H-" | cut -f3- > ${INPUT}.op
