#! /bin/bash

set -e

VOCAB_FILE=vocab/chinese_vocab.model

# DATA_PATH=data/outgen/train.jsonl
# SAVE_PATH=data/outgen/train-story.json

DATA_PATH=data/storal_zh/train.jsonl
SAVE_PATH=data/storal_zh/train-moral-story.json

# TYPE=post-train
TYPE=fine-tune


python -u preprocess.py \
    --vocab_file ${VOCAB_FILE} \
    --data_path ${DATA_PATH} \
    --save_path ${SAVE_PATH} \
    --type ${TYPE}
