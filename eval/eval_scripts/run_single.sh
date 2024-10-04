CKPT_PATH=/data2/zhangyu/ds-vqa-output/DATE04-25_Epoch6_LR1e-3_data_lsvq_align_all_1
CKPT_NAMES="epoch-5 epoch-4 epoch-3 epoch-2 epoch-1 epoch-0"
VISION_MODEL=/data/zhangyu/own_data/public_models/models--openai--clip-vit-large-patch14
LLM=/data2/zhaoxin/pretrain_models/llama3_8b_instruct

EVAL_DATSET=LSVQ_whole_test_1080p_ds_class

SAVE_PATH=results/${EVAL_DATSET}
EVAL_ROOT=/data/zhangyu/own_data/VQA/LSVQ/LSVQ
FEATURE_ROOT=/data1/zhangyu/own_data/VQA/LSVQ/LSVQ_feature_1fps
MOTION_ROOT=/data1/zhangyu/own_data/VQA/LSVQ/LSVQ_SlowFast2_feature
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

