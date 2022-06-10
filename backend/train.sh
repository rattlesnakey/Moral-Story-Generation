#! /bin/bash


set -e

echo 'post-training...'
DEVICE=0,1,2
VOCAB_FILE=vocab/chinese_vocab.model
MODEL_CONFIG=config/cpm-small.json
TRAIN_DATA_PATH=data/story/outgen/train-story.json
EVAL_DATA_PATH=data/story/outgen/train-story.json
MAX_LEN=200
EPOCHS=3
BATCH_SIZE=5
EVAL_BATCH_SIZE=2
GPU0_BATCH_SIZE=5
LR=1.5e-4
ACCU_GRADIENT=1
SAVE_MODEL_PATH=model/story_${EPOCHS}_${LR}_${BATCH_SIZE}
PRETRAIN_MODEL_PATH=model/zuowen_epoch40
SEED=42
WARMUP=800
METRIC_SAVE_PATH=${SAVE_MODEL_PATH}/metric.json
NUM_WORKER=2
PROJECT_NAME=cpm-story-gen
ENTITY=hengyuan
TASK=post-training




python -u train.py \
    --device ${DEVICE} \
    --vocab_path ${VOCAB_FILE} \
    --model_config ${MODEL_CONFIG} \
    --train_path ${TRAIN_DATA_PATH} \
    --eval_path ${EVAL_DATA_PATH} \
    --max_len ${MAX_LEN} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --gpu0_bsz ${GPU0_BATCH_SIZE} \
    --lr ${LR} \
    --gradient_accumulation_steps ${ACCU_GRADIENT} \
    --save_model_path ${SAVE_MODEL_PATH} \
    --pretrained_model ${PRETRAIN_MODEL_PATH} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKER} \
    --warmup_steps ${WARMUP} \
    --metric_save_path ${METRIC_SAVE_PATH} \
    --task ${TASK} \
    --project_name ${PROJECT_NAME} \
    --entity ${ENTITY}



wait

echo 'fine-tuning....'

TRAIN_DATA_PATH=data/moral-story/train-moral-story.json
EVAL_DATA_PATH=data/moral-story/train-moral-story.json
PRETRAIN_MODEL_PATH=${SAVE_MODEL_PATH}/best
SAVE_MODEL_PATH=model/moral-story_${EPOCHS}_${LR}_${BATCH_SIZE}
METRIC_SAVE_PATH=${SAVE_MODEL_PATH}/metric.json
TASK=fine-tuning

python -u train.py \
    --device ${DEVICE} \
    --vocab_path ${VOCAB_FILE} \
    --model_config ${MODEL_CONFIG} \
    --train_path ${TRAIN_DATA_PATH} \
    --eval_path ${EVAL_DATA_PATH} \
    --max_len ${MAX_LEN} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --gpu0_bsz ${GPU0_BATCH_SIZE} \
    --lr ${LR} \
    --gradient_accumulation_steps ${ACCU_GRADIENT} \
    --save_model_path ${SAVE_MODEL_PATH} \
    --pretrained_model ${PRETRAIN_MODEL_PATH} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKER} \
    --warmup_steps ${WARMUP} \
    --metric_save_path ${METRIC_SAVE_PATH} \
    --task ${TASK} \
    --project_name ${PROJECT_NAME} \
    --entity ${ENTITY}

