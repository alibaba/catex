#!/bin/bash

# custom config
DATA=./data
TRAINER=CATEX
SHOTS=-1  # we use all samples
NCTX=16
CSC=True
CTP=end

epoch=50

DATASET=${1:-'imagenet100_mcm'}
CFG=${2:-'vit_b16_ep50'}
MODELDIR=${3:-'weights/imagenet100-MCM'}

CTX_INIT=${4:-None}
OOD_PROMPT_NUM=${5:-1}
OOD_INFER_INTEGRATE=${6:-True}
OOD_INFER_OPTION=${7:-None}

FEAT_AS_INPUT=${8:-True}

for SEED in 1 #2 3
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir ${MODELDIR}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch ${epoch} \
    --eval-only \
    --ood-test \
    TRAINER.CATEX.N_CTX ${NCTX} \
    TRAINER.CATEX.CSC ${CSC} \
    TRAINER.CATEX.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.CATEX.CTX_INIT ${CTX_INIT} \
    TRAINER.OOD_PROMPT True \
    TRAINER.OOD_PROMPT_NUM ${OOD_PROMPT_NUM} \
    TRAINER.OOD_INFER 'MCM' \
    TRAINER.OOD_INFER_INTEGRATE ${OOD_INFER_INTEGRATE} \
    TRAINER.OOD_INFER_OPTION ${OOD_INFER_OPTION} \
    TRAINER.FEAT_AS_INPUT ${FEAT_AS_INPUT} \
    TRAINER.ID_FEAT_PRELOAD ''
done
