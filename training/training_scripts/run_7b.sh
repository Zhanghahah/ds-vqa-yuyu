


# path to pretrain models
VISION_MODEL=/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14
LLM=/data2/zhaoxin/pretrain_models/llama3_8b_instruct 
##for finetune:  LLM=output/DATE01-19_Epoch6_LR1e-3_data_lsvq_align_all_1/epoch-5


EPOCH=3
ZERO_STAGE=3
lr=1e-3
DATA_PATH=data/video_score_data

DATE=07-03
DATA="lsvq_align"  #  youtube_ugc, 1K, live_vqc

DATA_SAMPLE="all"
#IMAGE_PER_SAMPLE="3 2 1 1 1 1 1 1 1"
IMAGE_PER_SAMPLE="1"


DATA_CONCATE="${DATA// /_}"
DATA_SAMPLE_CONCATE="${DATA_SAMPLE// /_}"
IMAGE_CONCATE="${IMAGE_PER_SAMPLE// /_}"
# 

OUTPUT_Base=output/

OUTPUT_Dir=DATE${DATE}_Epoch${EPOCH}_LR${lr}_data_${DATA_CONCATE}_${DATA_SAMPLE_CONCATE}_${IMAGE_CONCATE}

OUTPUT=${OUTPUT_Base}${OUTPUT_Dir}

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT
mkdir -p ./log/$OUTPUT_Dir/

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
deepspeed main.py --max_seq_len 4096 \
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
    --pre_output_dir $OUTPUT \
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