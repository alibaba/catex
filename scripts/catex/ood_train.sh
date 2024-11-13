#!/bin/bash

# custom config
DATA=./data
TRAINER=CATEX
SHOTS=-1  # number of shots (1, 2, 4, 8, 16, -1). -1 means we use all samples
NCTX=16  # number of context tokens
CSC=True  # class-specific context (False or True)
CTP=end  # class token position (end or middle)

epoch=50

DATASET=${1:-'imagenet'}
CFG=${2:-'vit_b16_ep50'}

ID_PERTUR_NUM=${3:-1}
OOD_PROMPT_NUM=${4:-1}
OOD_PROMPT_ORTH=${5:-False}

FEAT_AS_INPUT=${6:-False}

for SEED in 1 #2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --ood-train \
        TRAIN.PRINT_FREQ 100 \
        TRAIN.CHECKPOINT_FREQ 5 \
        TRAINER.CATEX.N_CTX ${NCTX} \
        TRAINER.CATEX.CSC ${CSC} \
        TRAINER.CATEX.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.OOD_OE_LOSS False \
        TRAINER.OOD_PROMPT True \
        TRAINER.OOD_PROMPT_CE_LOSS True \
        TRAINER.OOD_PROMPT_MARGIN_LOSS True \
        TRAINER.OOD_PROMPT_MARGIN_SOFT_LOSS True \
        TRAINER.OOD_ANCHOR True \
        TRAINER.ID_PERTURB_LOSS False \
        TRAINER.ID_FEAT_PRELOAD '' \
        TRAINER.FEAT_AS_INPUT ${FEAT_AS_INPUT} \
        TRAINER.START_EPOCH 3 \
        TRAINER.ID_PERTUR_NUM ${ID_PERTUR_NUM} \
        TRAINER.OOD_PROMPT_NUM ${OOD_PROMPT_NUM} \
        TRAINER.OOD_PROMPT_ORTH ${OOD_PROMPT_ORTH}
    fi
done
