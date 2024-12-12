
VISION_MODEL=openai/clip-vit-large-patch14

EVAL_DATSET=youtube_ugc_ds # file name of metadata

SAVE_PATH=results/${EVAL_DATSET}
mkdir -p ${SAVE_PATH}

#NOTE: to run multi-GPU, you simple do "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"
export CUDA_VISIBLE_DEVICES=0; python3 generation_extract_feature.py --vis_proj vit \
    --vision_model_name_or_path ${VISION_MODEL} \
    --save_path data/video_score_data/youtube_ugc/youtube_ugc_feature_1fps \
    --eval_data ${EVAL_DATSET} \
    --eval_root data/video_score_data/youtube_ugc/youtube_ugc_videos \
    --video_loader_type decord > ${SAVE_PATH}/ours_infer.log


