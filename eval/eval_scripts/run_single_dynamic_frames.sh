#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# 1K: KoNViD_1k_features
#EVAL_DATSET=eval_robustness  eval_single eval_comprehensive (see the json in the folder ./eval_data/*.json)
CKPT_PATH=/data/zhangyu/ds-vqa-yuyu/training/output/DATE02-19_Epoch10_LR1e-3_data_lsvq_align_all_1
CKPT_NAMES="epoch-6 epoch-7 epoch-8"
VISION_MODEL=/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14
LLM=/data/zhangyu/own_data/public_models/Llama-2-7b-hf #meta-llama/Llama-2-7b
EVAL_DATSET=YT-ugc_train_ds
# LSVQ_whole_test_1080p_ds_score
# 1K: Konvid-1k_test_ds_repaired; Konvid-1k_train_ds
# ugc: YT-ugc_test_ds
# live vqc: LIVE-VQC_test_ds LIVE-VQC_train_ds

SAVE_PATH=results/${EVAL_DATSET}
mkdir -p ${SAVE_PATH}

#NOTE: to run multi-GPU, you simple do "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"
export CUDA_VISIBLE_DEVICES=0; python3 generation_dynamic_frames.py --model_name dsvl --vis_proj vit --max_seq_len 4096 \
    --eval_root /data1/zhangyu/own_data/VQA/youtube_ugc/h264 \
    --eval_feat /data1/zhangyu/own_data/VQA/youtube_ugc/youtube_ugc_feature_1fps \
    --lm_model_name_or_path  ${LLM} --vision_model_name_or_path ${VISION_MODEL} \
    --checkpoint_path $CKPT_PATH  --checkpoint_names $CKPT_NAMES --eval_data ${EVAL_DATSET} \
    --enable_mmca_attention --output_filename ${SAVE_PATH} > ${SAVE_PATH}/ours_infer.log

