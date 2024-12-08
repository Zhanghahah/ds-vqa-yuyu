#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

#EVAL_DATSET=eval_robustness  eval_single eval_comprehensive (see the json in the folder ./eval_data/*.json)
# CKPT_PATH=/data/zhangyu/ds-vqa-yuyu/training/output/DATE02-17_Epoch6_LR1e-3_data_lsvq_align_all_1
CKPT_PATH=/data1/zhangyu/DATE02-17_Epoch6_LR1e-3_data_lsvq_align_all_1
CKPT_NAMES="epoch-5"
VISION_MODEL=/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14


EVAL_DATSET=LBVD_ds

SAVE_PATH=results/${EVAL_DATSET}
mkdir -p ${SAVE_PATH}

#NOTE: to run multi-GPU, you simple do "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"
export CUDA_VISIBLE_DEVICES=0; python3 generation_extract_feature.py --model_name dsvl --vis_proj vit \
    --vision_model_name_or_path ${VISION_MODEL} \
    --save_path /data1/zhangyu/own_data/VQA/LBVD/LBVD_feature_1fps\
    --checkpoint_path $CKPT_PATH  \
    --checkpoint_names $CKPT_NAMES \
    --eval_data ${EVAL_DATSET} \
    --eval_root /data1/zhangyu/own_data/VQA/LBVD/videos\
    --video_loader_type decord \
    --output_filename ${SAVE_PATH} > ${SAVE_PATH}/ours_infer.log



            # "args": [
            #     "--model_name",
            #     "dsvl",
            #     "--vis_proj",
            #     "vit",
            #     "--max_seq_len",
            #     "4096",
            #     "--lm_model_name_or_path",
            #     "/data/zhangyu/own_data/public_models/Llama-2-7b-hf",
            #     "--vision_model_name_or_path",
            #     "/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14",
            #     "--checkpoint_path",
            #     "/data1/zhangyu/DATE02-17_Epoch6_LR1e-3_data_lsvq_align_all_1",
            #     "--checkpoint_names",
            #     "epoch-5",
            #     "--eval_data",
            #     "LBVD_ds",
            #     "--enable_mmca_attention",
            #     "--output_filename",
            #     "results/LBVD_ds_test",
            #     "--save_path",
            #     "/data1/zhangyu/own_data/VQA/debug/LBVD_feature_1fps",
            #     "--eval_root",
            #     "/data1/zhangyu/own_data/VQA/LBVD/videos",
            # ]

