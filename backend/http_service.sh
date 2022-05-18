#! /bin/bash
set -e

PORT=8085
MODEL_PATH=model/moral-story_100_1.5e-4_32/best
CONTEXT_LEN=200

python http_service.py \
    --port ${PORT} \
    --model_path ${MODEL_PATH} \
    --context_len ${CONTEXT_LEN}