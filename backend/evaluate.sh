#! /bin/bash
set -e

MODEL_PATH=model/zuowen_epoch40
CONTEXT_LEN=200
TEST_DATASET_PATH=data/story/test.jsonl
METRIC_OUTPUT_PATH=model/zuowen_epoch40
TASK=post-training

python eval_metric.py \
    --model_path ${MODEL_PATH} \
    --test_dataset_path ${TEST_DATASET_PATH} \
    --context_len ${CONTEXT_LEN} \
    --metric_output_path ${METRIC_OUTPUT_PATH} \
    --task ${TASK}