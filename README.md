## Introduction
We propose the first large multi-modal based video quality assessment (LMM-VQA) model, which introduces a novel spatiotemporal visual modeling strategy for quality-aware feature extraction. Specifically, we first reformulate the quality regression problem into a question and answering (Q\&A) task and construct Q\&A prompts for VQA instruction tuning. Then, we design a spatiotemporal vision encoder to extract spatial and temporal features to represent the quality characteristics of videos, which are subsequently mapped into the language space by the spatiotemporal projector for modality alignment. Finally, the aligned visual tokens and the quality-inquired text tokens are aggregated as inputs for the large language model to generate the quality score as well as the quality level. Extensive experiments demonstrate that LMM-VQA achieves state-of-the-art performance across five VQA benchmarks, exhibiting an average improvement of 5% in generalization ability over existing methods. Furthermore, due to the advanced design of the spatiotemporal encoder and projector, LMM-VQA also performs exceptionally well on general video understanding tasks, further validating its effectiveness.

<p align="center">
<img src="./assets/framework8_.pdf" height = "360" alt="" align=center />
</p>

## Installation Guide
Create a virtual environment to run the code of LMM-VQA.<br>
```
cd ds-vqa-yuyu
conda create -n llm-vqa python=3.9
conda activate llm-vqa
pip install -r requirements.txt
```

## Data preparation

### Download pretrain weights and databases

[llama3_8b_instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B),                [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)


[LSVQ](https://github.com/baidut/PatchVQ),
[KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html),
[Youtube-UGC](https://media.withyoutube.com/),
[LIVE-VQC](https://live.ece.utexas.edu/research/LIVEVQC/index.html),
[Youtube-Gaming](https://live.ece.utexas.edu/research/LIVE-YT-Gaming/index.html)


### Prompt data

`data/prompt_data` contains the processed VQA data, and `data/question_prompt` contains the prompts. If you want to generate your own data, please refer to the paper and `preprocess_ds.py`.

## Usage

### Video feature extraction

1. Extract video spatial feature
```bash
bash eval/eval_scripts/run_extract_frame.sh
```

2. Extract motion feature

For LSVQ datasets:

```shell
 CUDA_VISIBLE_DEVICES=0 python -u extract_SlowFast_features_LSVQ.py \
 --database LSVQ \
 --model_name SlowFast \
 --resize 224 \
 --feature_save_folder LSVQ_SlowFast_feature/ \
 --videos_dir /data1/own_data/VQA/LSVQ/LSVQ \
 >> logs/extract_SlowFast_features_LSVQ.log
```

  
For KoNViD-1k, Youtube-UGC, LIVE-VQC, and Youtube-Gaming datasets:

```shell
 CUDA_VISIBLE_DEVICES=0 python -u python extract_motion_VQA.py \
 --database KoNViD-1k \
 --model_name SlowFast \
 --resize 224 \
 --feature_save_folder KoNViD-1k_SlowFast_feature/ \
 --videos_dir /data1/own_data/VQA/youtube_ugc/h264 \
 >> logs/extract_motion_VQA.log
```

### Training

We provide experiment scripts under the folder `training/training_scripts`. For instance, you can train the model with text and videos by:

```bash
bash training/training_scripts/run_7b.sh
```

### Inference

```bash
bash eval/eval_scripts/run_single.sh 
```

### Evaluation

```bash
bash metric_csv.sh
```
