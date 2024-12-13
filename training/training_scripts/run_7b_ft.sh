#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


VISION_MODEL=public_models/models--openai--clip-vit-large-patch14
LLM=pretrain_models/llama3_8b_instruct




EPOCH=10
PRE_EPOCH=6
ZERO_STAGE=3
lr=1e-3

PRE_DATE=04-25
DATE=06-26-ft-live_vqc
DATA_PATH=data
#DATA="llava llava_dial otter_mimicit_cgd otter_mimicit_sd otter_mimicit_sn otter_mimicit_tvc otter_mimicit_vst llava_otter_blend sparkles_dialogue"
DATA="live_vqc"
PRE_DATA="lsvq_align"
DATA_SAMPLE="all"
#IMAGE_PER_SAMPLE="3 2 1 1 1 1 1 1 1"
IMAGE_PER_SAMPLE="1"


DATA_CONCATE="${DATA// /_}"
PRE_DATA_CONCATE="${PRE_DATA// /_}"
DATA_SAMPLE_CONCATE="${DATA_SAMPLE// /_}"
IMAGE_CONCATE="${IMAGE_PER_SAMPLE// /_}"
#

OUTPUT_Base=ds-vqa-output/

PRE_OUTPUT_Dir=DATE${PRE_DATE}_Epoch${PRE_EPOCH}_LR${lr}_data_${PRE_DATA_CONCATE}_${DATA_SAMPLE_CONCATE}_${IMAGE_CONCATE}
OUTPUT_Dir=DATE${DATE}_Epoch${EPOCH}_LR${lr}_data_${DATA_CONCATE}_${DATA_SAMPLE_CONCATE}_${IMAGE_CONCATE}

PRE_OUTPUT=${OUTPUT_Base}${PRE_OUTPUT_Dir}
OUTPUT=${OUTPUT_Base}${OUTPUT_Dir}


if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT
mkdir -p ./log/$OUTPUT_Dir/

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
python3 main.py --max_seq_len 4096 \
    --data_path ${DATA_PATH} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 8 \
    --lm_model_name_or_path  ${LLM} \
    --vision_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing --vis_proj vit \
    --save_img_feature \
    --motion_token \
    --video_loader_type opencv \
    --precision bf16 \
    --gradient_accumulation_steps 1  --zero_stage $ZERO_STAGE --learning_rate $lr --num_warmup_steps 0.1 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 2 --deepspeed \
    --output_dir $OUTPUT \
    --pre_output_dir $PRE_OUTPUT \
    --num_train_epochs ${EPOCH} --enable_mmca_attention --enable_tensorboard \
    > $OUTPUT/training.log
# single gpu
#python3 main.py --max_seq_len 4096 \
#    --data_path ${DATA_PATH} \
#    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 8 \
#    --lm_model_name_or_path  ${LLM} \
#    --vision_model_name_or_path ${VISION_MODEL} \
#    --gradient_checkpointing --vis_proj vit \
#    --video_loader_type decord \
#    --gradient_accumulation_steps 1  --zero_stage $ZERO_STAGE --learning_rate $lr --num_warmup_steps 0.1 \
#    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --deepspeed --output_dir $OUTPUT  --num_train_epochs ${EPOCH} --enable_mmca_attention --enable_tensorboard \
#    > $OUTPUT/training.log