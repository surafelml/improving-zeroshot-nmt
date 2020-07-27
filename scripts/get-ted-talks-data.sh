#!/bin/bash

set -e

EXPDIR=$PWD	# update accordingly 
DATA=$EXPDIR/data
mkdir -p $DATA/ted-data

pushd $DATA
  wget http://phontron.com/data/ted_talks.tar.gz #-P ted_data/
  tar -xzvf ted_talks.tar.gz -C ted-data/
  rm -rf ted_talks.tar.gz

  # we use ./scripts/ted_talks_langs.txt to extract all langs-en pairs to ./data/ted_data
  # feeding the script lang ids. https://github.com/neulab/word-embeddings-for-nmt/blob/master/ted_reader.py
  python $EXPDIR/scripts/ted_reader.py

  rm -rf ./ted_data/*.tsv
popd
