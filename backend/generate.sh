#! /bin/bash
set -e

TOPK=20
TOPP=0.85
TEMPERATURE=0.8
MODEL_PATH=model/moral-story_100_1.5e-4_32/best
DEVICE=0
REPETITION_PENALTY=1.1

python -u generate.py \
    --topk ${TOPK} \
    --topp ${TOPP} \
    --temperature ${TEMPERATURE} \
    --repetition_penalty ${REPETITION_PENALTY} \
    --no_cuda 
