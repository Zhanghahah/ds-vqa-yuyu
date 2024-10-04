#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# This file is for evaluation of 1K datasets

#EVAL_DATSET=eval_robustness  eval_single eval_comprehensive (see the json in the folder ./eval_data/*.json)
CKPT_PATH=/data2/zhangyu/ds-vqa-output/DATE06-20-ft-YT-ugc_Epoch10_LR1e-3_data_kvq_all_1
CKPT_NAMES="epoch-6 epoch-7 epoch-8 epoch-9"
VISION_MODEL=/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14
#VISION_MODEL=/data/zhangyu/ds-vqa-yuyu/helper/qwen_clip
LLM=/data2/zhaoxin/pretrain_models/llama3_8b_instruct
#/data/zhangyu/own_data/public_models/Llama-2-7b-hf
# /data/zhaoxin/pretrain_models/llama3_8b_instruct
EVAL_DATSET=YT-ugc_ds
#Konvid-1k_test_ds_repaired LSVQ_whole_test_1080p_ds_score LSVQ_whole_test_ds_score

SAVE_PATH=results/${EVAL_DATSET}
EVAL_ROOT=/data1/zhangyu/own_data/VQA/youtube_ugc/h264
FEATURE_ROOT=/data1/zhangyu/own_data/VQA/youtube_ugc/youtube_ugc_feature_1fps
MOTION_ROOT=/data1/zhangyu/own_data/VQA/youtube_ugc/youtube_ugc_SlowFast2_feature
mkdir -p ${SAVE_PATH}

#NOTE: to run multi-GPU, you simple do "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"
export CUDA_VISIBLE_DEVICES=1; python batch_generation_video.py --model_name dsvl --vis_proj vit --max_seq_len 4096 \
    --lm_model_name_or_path  ${LLM} --vision_model_name_or_path ${VISION_MODEL} \
    --checkpoint_path $CKPT_PATH  --checkpoint_names $CKPT_NAMES --eval_data ${EVAL_DATSET} \
    --eval_root ${EVAL_ROOT} \
    --feature_root ${FEATURE_ROOT} \
    --motion_root ${MOTION_ROOT} \
    --use_motion_token \
    --use_img_feature \
    --enable_mmca_attention --output_filename ${SAVE_PATH} > ${SAVE_PATH}/ours_infer.log

