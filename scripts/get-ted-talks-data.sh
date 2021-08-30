#!/bin/bash

set -e

EXPDIR=$PWD	# update accordingly
DATA=$EXPDIR/data
mkdir -p $DATA/ted-data

pushd $DATA
  wget http://phontron.com/data/ted_talks.tar.gz #-P ted_data/
  tar -xzvf ted_talks.tar.gz -C ted-data/
  rm -rf ted_talks.tar.gz

  # use the pre-specified src-trg lang-pairs from ./scripts/ted_talks_langs.txt to extract parallel data from ./ted_data
  # for pairs other than it/ro-en, update ./scripts/ted_talks_langs.txt
  python $EXPDIR/scripts/ted_reader.py --lang-file $EXPDIR/scripts/ted_talks_langs.txt --data-path ./ted-data/

  rm -rf ./ted_data/*.tsv
popd
