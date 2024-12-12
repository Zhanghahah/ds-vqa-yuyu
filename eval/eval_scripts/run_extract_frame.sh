#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

#EVAL_DATSET=eval_robustness  eval_single eval_comprehensive (see the json in the folder ./eval_data/*.json)
# CKPT_PATH=/data/zhangyu/ds-vqa-yuyu/training/output/DATE02-17_Epoch6_LR1e-3_data_lsvq_align_all_1
# CKPT_PATH=/data1/zhangyu/DATE02-17_Epoch6_LR1e-3_data_lsvq_align_all_1
# CKPT_NAMES="epoch-5"
VISION_MODEL=/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14

# /data2/zhangyu/own_data/VQA/zhichao/aigc_video/feature_1fps
EVAL_DATSET=aigc_ds_score

# SAVE_PATH=results/${EVAL_DATSET}
# mkdir -p ${SAVE_PATH}

#NOTE: to run multi-GPU, you simple do "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"
export CUDA_VISIBLE_DEVICES=0; python3 generation_extract_feature.py --vis_proj vit \
    --vision_model_name_or_path ${VISION_MODEL} \
    --save_path /data2/zhangyu/own_data/VQA/zhichao/aigc_video/feature_1fps\
    --eval_data ${EVAL_DATSET} \
    --eval_root /data1/zhangyu/own_data/VQA/zhichao/aigc_video/videos\
    --video_loader_type decord > ${SAVE_PATH}/ours_infer.log



# --save_path /data1/zhangyu/own_data/VQA/LBVD/LBVD_feature_1fps\
# --eval_root /data2/zhangyu/own_data/VQA/1K/KoNViD_1k_videos\
