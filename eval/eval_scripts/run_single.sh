#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

#EVAL_DATSET=eval_robustness  eval_single eval_comprehensive (see the json in the folder ./eval_data/*.json)
CKPT_PATH=/data/zhangyu/DeepSpeedExamples/applications/DeepSpeed-VisualChat/training/output/Epoch6_LR1e-3_data_lsvq_all_1
CKPT_NAME=epoch-5
VISION_MODEL=/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14
LLM=/data/zhangyu/own_data/public_models/Llama-2-7b-hf #meta-llama/Llama-2-7b
EVAL_DATSET=eval_lsvq

SAVE_PATH=eval/results/${EVAL_DATSET}
mkdir ${SAVE_PATH}
#NOTE: to run multi-GPU, you simple do "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"
export CUDA_VISIBLE_DEVICES=0; python eval/batch_generation.py --model_name dsvl --vis_proj baseline --max_seq_len 4096 \
    --lm_model_name_or_path  ${LLM} --vision_model_name_or_path ${VISION_MODEL} \
    --checkpoint_path $CKPT_PATH  --checkpoint_names $CKPT_NAME --eval_data ${EVAL_DATSET} \
    --enable_mmca_attention --output_filename ${SAVE_PATH}/ours_${CKPT_NAME} &> ${SAVE_PATH}/ours_${CKPT_NAME}.log

