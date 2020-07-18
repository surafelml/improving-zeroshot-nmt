#!/bin/bash 
set -e

SRC=$1		# src 
TGT=$2		# tgt
GPUS=$3		# gpu id, -1 for cpu

EXPDIR=$PWD/pretrain-model	# baseline model pre-training for zst-model 
DATA=$EXPDIR/bin-data		
MODEL=$EXPDIR/model		
mkdir -p $MODEL
LOG=$EXPDIR/log.train


CUDA_VISIBLE_DEVICES=$GPUS fairseq-train $DATA -s $SRC -t $TGT \
			--max-epoch 50 \
    			--optimizer adam --lr 0.0005 \
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
			--dropout 0.1 \
			--attention-dropout 0.1 \
			--share-all-embeddings \
			--no-epoch-checkpoints \
			--validate-interval 5 \
			--save-dir $MODEL \
			> $LOG 2> $LOG

echo "TRAINING LOG: $LOG"
# Note: for optimal performance update config according to the experimental setting.
