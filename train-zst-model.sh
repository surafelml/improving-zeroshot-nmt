#!/bin/bash

set -e

# === UPDATE ACCORDINGLY ===
EXPDIR=$PWD
SRC=$1					# it 		
TGT=$2					# ro
PRETRAINING=$3				# $EXPDIR/pretrain-baseline
ROUNDS=$4				# maximum zst modeling rounds, each round includes an inference ed training with a single pass on data
GPUS=$5					# gpu-id, -1 for cpu

# (optional) configs
PIVOT="en"				# follows the established assumption of a Pivot/P language, 
ADD_LANGID=true
MERGE_ZST_NONZST=true 			# merge zst pseudo-parallel with parallel data
MAX_EPOCH=1				# num of epcoh to train in each round

# generic settings
DATA=$EXPDIR/data/ted-data

# pretrained model 
PRE_DATA=$PRETRAINING/data/pre-data 	# as monolingual we use the X and Y side of X-P and Y-P parallel pairs 
BPE_MODEL=$PRETRAINING/data/bpe-model/code.32000.bpe
BPE_DATA=$PRETRAINING/data/bpe-data
TRAIN_SRC=$BPE_DATA/train.src
TRAIN_TGT=$BPE_DATA/train.tgt
BIN_DATA=$PRETRAINING/data/bin-data
PRETRAINED_MODEL=$PRETRAINING/model/checkpoint_best.pt

# zst model 
ZSTMODELING=$EXPDIR/zst-modeling
ZST_MODEL=$ZSTMODELING/model
ZST_CKPT=$ZST_MODEL/checkpoint_last.pt
DATA_ZST=$ZSTMODELING/data
mkdir -p $DATA_ZST
BIN_DATA_ZST=$ZSTMODELING/bin-data
LOG=$ZSTMODELING/log.train
LOG_EVAL=$ZSTMODELING/log.eval

# mono ed parallel eval data for zst
MONO_SRC=$DATA/$SRC-$PIVOT/train.$SRC
MONO_TGT=$DATA/$TGT-$PIVOT/train.$TGT
EVALDATA=$DATA/$SRC-$PIVOT-$TGT
DEV_SRC=$EVALDATA/dev.$SRC
DEV_TGT=$EVALDATA/dev.$TGT
TEST_SRC=$EVALDATA/test.$SRC
TEST_TGT=$EVALDATA/test.$TGT

# data created ed/or consumed at each zst modeling rounds 
MONO_SRC_TOK=$DATA_ZST/mono.src 
MONO_SRC_BPE=$DATA_ZST/mono.bpe.src
MONO_SRC_INFER=$DATA_ZST/mono.infer.src 
MONO_SRC_INFER_BPE=$DATA_ZST/mono.infer.bpe.src 

MONO_TGT_TOK=$DATA_ZST/mono.tgt
MONO_TGT_BPE=$DATA_ZST/mono.bpe.tgt
MONO_TGT_LANGID=$DATA_ZST/mono.langid.tgt 	# langid's for post inference preprocessing

PARA_DEV_SRC=$DATA_ZST/dev.$SRC			# (optional) dev set
PARA_DEV_SRC_TAG=$DATA_ZST/dev.tag.$SRC
PARA_DEV_TGT=$DATA_ZST/dev.$TGT
PARA_DEV_TGT_TAG=$DATA_ZST/dev.tag.$TGT

PARA_TEST_SRC=$DATA_ZST/test.$SRC
PARA_TEST_SRC_TAG=$DATA_ZST/test.tag.$SRC
PARA_TEST_TGT=$DATA_ZST/test.$TGT
PARA_TEST_TGT_TAG=$DATA_ZST/test.tag.$TGT

#PARA_DEV_SRC_ALL=$DATA_ZST/dev.src 		# (optional)
#PARA_DEV_TGT_ALL=$DATA_ZST/dev.tgt

PARA_TRAIN_SRC=$DATA_ZST/train.src
PARA_TRAIN_TGT=$DATA_ZST/train.tgt


# scripts 
MOSES=$EXPDIR/mosesdecoder/scripts
NORM=$MOSES/tokenizer/normalize-punctuation.perl
TOK=$MOSES/tokenizer/tokenizer.perl
DEES=$MOSES//tokenizer/deescape-special-chars.perl
MULTIBLUE=$MOSES/generic/multi-bleu.perl

INFERENCE=$EXPDIR/inference.sh
CBLEU=$EXPDIR/scripts/compute-bleu.sh
# ===	===



# preprocess monolingual data for dual zst directions
if [ ! -f $MONO_SRC_TOK ]; then
  echo "PREPROCESSING [$SRC <> $TGT] ZST DIRECTIONS MONOLINGUL DATA..."
  $NORM < $MONO_SRC | $TOK -l $SRC -q | $DEES | awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag" "$0 }' > $MONO_SRC_TOK
  $NORM < $MONO_TGT | $TOK -l $TGT -q | $DEES | awk -vtgt_tag="<2${SRC}>" '{ print tgt_tag" "$0 }' >> $MONO_SRC_TOK

  subword-nmt apply-bpe -c $BPE_MODEL < $MONO_SRC_TOK > $MONO_SRC_BPE
fi 

if [ ! -f $MONO_TGT_LANGID ]; then 
  # save only tags for post inference use
  $NORM < $MONO_SRC | $TOK -l $SRC -q | $DEES | awk -vtgt_tag="<2${SRC}>" '{ print tgt_tag }' > $MONO_TGT_LANGID
  $NORM < $MONO_TGT | $TOK -l $TGT -q | $DEES | awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag }' >> $MONO_TGT_LANGID
fi

if [ ! -f $MONO_TGT_TOK ]; then
  $NORM < $MONO_SRC | $TOK -l $SRC -q | $DEES > $MONO_TGT_TOK
  $NORM < $MONO_TGT | $TOK -l $TGT -q | $DEES >> $MONO_TGT_TOK

  subword-nmt apply-bpe -c $BPE_MODEL < $MONO_TGT_TOK > $MONO_TGT_BPE
fi


# preprocess parallel evaluation data for zst directions 
if [ ! -f $PARA_DEV_SRC_TAG  ] || [ ! -f $PARA_DEV_TGT_TAG ]; then 
  echo "PREPROCESSING [$SRC <-> $TGT] PARALLEL DEV SET ..."
  $NORM < $DEV_SRC | $TOK -l $SRC -q | $DEES | awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag" "$0 }' | subword-nmt apply-bpe -c $BPE_MODEL > $PARA_DEV_SRC_TAG
  $NORM < $DEV_TGT | $TOK -l $TGT -q | $DEES | awk -vtgt_tag="<2${SRC}>" '{ print tgt_tag" "$0 }' | subword-nmt apply-bpe -c $BPE_MODEL > $PARA_DEV_TGT_TAG
  #subword-nmt apply-bpe -c $BPE_MODEL < ${PARA_TEST_SRC_TAG}.tmp > $PARA_TEST_SRC_TAG

  $NORM < $DEV_SRC | $TOK -l $SRC -q | $DEES  > $PARA_DEV_SRC
  $NORM < $DEV_TGT | $TOK -l $TGT -q | $DEES  > $PARA_DEV_TGT
fi

if [ ! -f $PARA_TEST_SRC_TAG  ] || [ ! -f $PARA_TEST_TGT_TAG ]; then 
  echo "PREPROCESSING [$SRC <> $TGT] PARALLEL TEST SET ..."
  $NORM < $TEST_SRC | $TOK -l $SRC -q | $DEES | awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag" "$0 }' | subword-nmt apply-bpe -c $BPE_MODEL > $PARA_TEST_SRC_TAG
  $NORM < $TEST_TGT | $TOK -l $TGT -q | $DEES | awk -vtgt_tag="<2${SRC}>" '{ print tgt_tag" "$0 }' | subword-nmt apply-bpe -c $BPE_MODEL > $PARA_TEST_TGT_TAG

  $NORM < $TEST_SRC | $TOK -l $SRC -q | $DEES  > $PARA_TEST_SRC
  $NORM < $TEST_TGT | $TOK -l $TGT -q | $DEES  > $PARA_TEST_TGT
fi



# ZT INFER-TRAIN LOOP 
for ROUND in $( seq $ROUNDS ); do 
	echo "ZST INFERENCE-TRAINING ROUND: [$ROUND]"

	# get pre-trained model 
	if [ ! -d $ZST_MODEL -a $ROUND = 1 ]; then 
	  mkdir $ZST_MODEL
	  echo $ZST_MODEL
	  cp $PRETRAINED_MODEL $ZST_CKPT 
	fi

	# get pre-trained model dict
	if [ -d $BIN_DATA -a $ROUND = 1 ]; then
	  mkdir -p $BIN_DATA_ZST
	  cp $BIN_DATA/dict.* $BIN_DATA_ZST/
	fi
	

     if [ $ROUND = 1 ]; then
	echo "PRE-EVALUATING TEST SET FOR ${SRC}<>${TGT} ..."
	bash $INFERENCE $BIN_DATA_ZST $PARA_TEST_SRC_TAG $ZST_CKPT $GPUS
	BLEU=$(bash $CBLEU ${PARA_TEST_SRC_TAG}.op ${PARA_TEST_TGT}) 
	echo "ROUND: 0 | ZST_DIR: $SRC-$TGT | MULTIBLEU=$BLEU" | tee -a $LOG_EVAL

	
	bash $INFERENCE $BIN_DATA_ZST $PARA_TEST_TGT_TAG $ZST_CKPT $GPUS
	BLEU=$(bash $CBLEU ${PARA_TEST_TGT_TAG}.op ${PARA_TEST_SRC}) 
	echo "ROUND: 0 | ZST_DIR: $TGT-$SRC | MULTIBLEU=$BLEU" | tee -a $LOG_EVAL
      fi	

	
	# zero-shot inference with monolingual data
	echo "ROUND [$ROUND] ZST MODELING INFERENCE ..."
	bash $INFERENCE $BIN_DATA_ZST $MONO_SRC_BPE $ZST_CKPT $GPUS
	wait $!
	

	# postprocess inference output with lang-id, and bpe
	mv $MONO_SRC_BPE  ${MONO_SRC_INFER}.tmp
	#mv ${MONO_SRC_INFER} ${MONO_SRC_INFER}.tmp
	paste -d' ' $MONO_TGT_LANGID ${MONO_SRC_INFER}.tmp > $MONO_SRC_INFER
	#rm -rf $MONO_SRC_INFER 
	subword-nmt apply-bpe -c $BPE_MODEL < $MONO_SRC_INFER > $MONO_SRC_INFER_BPE 

	
	# merge original pre-trained model data with new zst data 
	if $MERGE_ZST_NONZST; then
		cat $MONO_SRC_INFER_BPE $TRAIN_SRC >  $PARA_TRAIN_SRC

		if [ ! -f $PARA_TRAIN_TGT ]; then
		  cat $MONO_TGT_BPE $TRAIN_TGT >  $PARA_TRAIN_TGT
		fi
	else 	
		cat $MONO_SRC_INFER_BPE > $PARA_TRAIN_SRC

		if [ ! -f $PARA_TRAIN_TGT ]; then
		  cat $MONO_TGT_BPE >  $PARA_TRAIN_TGT
		fi
	fi


	# binarize (only source for round >= 2)
	if [ $ROUND = 1 ]; then
	  fairseq-preprocess -s src -t tgt \
				--destdir $BIN_DATA_ZST \
				--trainpref $DATA_ZST/train \
				--srcdict $BIN_DATA_ZST/dict.src.txt \
				--tgtdict $BIN_DATA_ZST/dict.tgt.txt \
				--workers 32 \
				>> $LOG 2>> $LOG
	else 
	  fairseq-preprocess -s src -t tgt \
				--destdir $BIN_DATA_ZST \
				--trainpref $DATA_ZST/train \
				--srcdict $BIN_DATA_ZST/dict.src.txt \
				--tgtdict $BIN_DATA_ZST/dict.tgt.txt \
				--only-source \
				--workers 32 \
				>> $LOG 2>> $LOG
	fi


	# train zst model
	CUDA_VISIBLE_DEVICES=$GPUS fairseq-train $BIN_DATA_ZST -s src -t tgt \
			--max-epoch $MAX_EPOCH \
    			--optimizer adam \
			--lr 0.0005 \
			--clip-norm 0.0 \
			--max-tokens 4096 \
			--no-progress-bar \
			--log-interval 100 \
			--min-lr '1e-09' \
			--weight-decay 0.0001 \
			--criterion label_smoothed_cross_entropy \
			--label-smoothing 0.1 \
			--lr-scheduler inverse_sqrt \
			--ddp-backend=no_c10d \
			--warmup-updates 4000 \
			--warmup-init-lr '1e-07' \
			--adam-betas '(0.9, 0.98)' \
			--arch transformer_iwslt_de_en \
			--dropout 0.3 \
			--attention-dropout 0.1 \
			--share-all-embeddings \
			--keep-last-epochs 1 \
			--no-epoch-checkpoints \
			--valid-subset 'train' \
			--disable-validation \
			--reset-dataloader \
			--reset-lr-scheduler \
			--reset-optimizer \
			--reset-meters \
			--save-dir $ZST_MODEL \
			>> $LOG 2>> $LOG 
			wait $!

			# Note, base on data size and keep in mind to adjust the at least the following params for best result
			# UPDATES 10K,100K
			# DATA SHUFFLE ?
			# DROPOUT ?
			# Original implementation includes (already included in fairseq) the following
			#--reset-lr-scheduler \
			#--reset-optimizer \
			#--reset-meters 


	
	echo "EVALUATING TEST SET FOR ${SRC}<>${TGT} ..."
	bash $INFERENCE $BIN_DATA_ZST $PARA_TEST_SRC_TAG $ZST_CKPT $GPUS
	BLEU=$(bash $CBLEU ${PARA_TEST_SRC_TAG}.op ${PARA_TEST_TGT}) 
	echo "ROUND: 0 | ZST_DIR: $SRC-$TGT | MULTIBLEU=$BLEU" | tee -a $LOG_EVAL

	bash $INFERENCE $BIN_DATA_ZST $PARA_TEST_TGT_TAG $ZST_CKPT $GPUS
	BLEU=$(bash $CBLEU ${PARA_TEST_TGT_TAG}.op ${PARA_TEST_SRC}) 
	echo "ROUND: 0 | ZST_DIR: $TGT-$SRC | MULTIBLEU=$BLEU" | tee -a $LOG_EVAL

done
echo "END OF ZST TRAINING AT ROUND: $ROUND ..."
