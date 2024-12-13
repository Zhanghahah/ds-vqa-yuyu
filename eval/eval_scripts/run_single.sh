
# path to ckpt
CKPT_PATH=output/DATE04-22_Epoch6_LR1e-3_data_lsvq_align_all_1
CKPT_NAMES="epoch-5 epoch-4"

#path to pretain models
VISION_MODEL=openai/clip-vit-large-patch14
LLM=meta/llama3_8b_instruct

#path to data
EVAL_DATSET=LSVQ_whole_test_ds_score
EVAL_ROOT=data/video_score_data/LSVQ/LSVQ_videos
FEATURE_ROOT=data/video_score_data/LSVQ/LSVQ_feature_1fps
MOTION_ROOT=data/video_score_data/LSVQ/LSVQ_SlowFast_feature

SAVE_PATH=results/${EVAL_DATSET}
mkdir -p ${SAVE_PATH}

export CUDA_VISIBLE_DEVICES=0; python batch_generation_video.py --model_name dsvl --vis_proj vit --max_seq_len 4096 \
    --lm_model_name_or_path  ${LLM} --vision_model_name_or_path ${VISION_MODEL} \
    --checkpoint_path $CKPT_PATH  --checkpoint_names $CKPT_NAMES --eval_data ${EVAL_DATSET} \
    --eval_root ${EVAL_ROOT} \
    --feature_root ${FEATURE_ROOT} \
    --motion_root ${MOTION_ROOT} \
    --use_motion_token \
    --use_img_feature \
    --enable_mmca_attention --output_filename ${SAVE_PATH} > ${SAVE_PATH}/ours_infer.log

