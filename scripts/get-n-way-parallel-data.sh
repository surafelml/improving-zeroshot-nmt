#!/bin/bash

set -e 

SRC=$1		# it
TGT=$2		# ro 
PIVOT=$3	# en


EXPDIR=$PWD
DATA=$EXPDIR/data/ted-data 		
DATA_SRC_PIVOT_TGT=$DATA/$SRC-$PIVOT-$TGT


if [ ! -d $DATA_SRC_PIVOT_TGT ]; then
  mkdir $DATA_SRC_PIVOT_TGT
  pushd $DATA_SRC_PIVOT_TGT
 
    for SET in dev test; do
	echo "GETTING [$SRC - $PIVOT - $TGT] PARALLEL [[ $SET ]] DATA..."
	
	  DATA_SRC=$DATA/$SRC-$PIVOT/${SET}
	  DATA_TGT=$DATA/$TGT-$PIVOT/${SET}

	  paste ${DATA_SRC}.${SRC} ${DATA_SRC}.${PIVOT} > ${SET}.${SRC}-${PIVOT}
	  paste ${DATA_TGT}.${TGT} ${DATA_TGT}.${PIVOT} > ${SET}.${TGT}-${PIVOT}

	  # get overlpaing src-pivot, -tgt segments 
	  awk -F'\t' 'NR==FNR{c[$2]++;next};c[$2] == 1' ${SET}.${SRC}-${PIVOT} ${SET}.${TGT}-${PIVOT} > ${SET}.$TGT-$PIVOT-overlap.$SRC-$PIVOT
	  awk -F'\t' 'NR==FNR{c[$2]++;next};c[$2] == 1' ${SET}.$TGT-$PIVOT-overlap.$SRC-$PIVOT ${SET}.${SRC}-${PIVOT} > ${SET}.$SRC-$PIVOT-overlap.$TGT-$PIVOT
	
	  # create three way parallel 
	  cut -f1 ${SET}.$TGT-$PIVOT-overlap.$SRC-$PIVOT > ${SET}.$TGT
	  cut -f2 ${SET}.$TGT-$PIVOT-overlap.$SRC-$PIVOT > ${SET}.$PIVOT
	  cut -f1 ${SET}.$SRC-$PIVOT-overlap.$TGT-$PIVOT > ${SET}.$SRC

	  rm -rf *overlap*  ${SET}.${SRC}-${PIVOT} ${SET}.${TGT}-${PIVOT} 
    done
  popd
fi
