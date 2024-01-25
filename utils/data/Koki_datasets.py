# This file is adapted from https://github.com/Zhanghahah/VQA-LLM
# This dataset is from lsvq

import os
import random
import decord
import cv2
import json
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from .vqa_dataset import VQADataset
import utils.data.DST as DST
from utils.utils import get_rank, print_rank_0
from .utils import save_debug_image, save_debug_text
from decord import VideoReader, cpu
from PIL import Image
from .lsvq_dataset import load_and_transform_video

import numpy as np
import torch

class KokiDataset(VQADataset):
    def __init__(self, data_path,
                 data_debug_path,
                 per_sample_image,
                 video_loader_type,
                 tokenizer,
                 vis_processor,
                 add_eos=True, ignore_instruction=True,save_video_feat=False, **kwargs):
        vis_root = f"{data_path}/1K/KoNViD_1k_videos"
        assert os.path.isdir(vis_root), f"KokiDataset image directory {vis_root} not found, you need to check the image path"

        feature_root = "/data/zhangyu/own_data/VQA/1K/KoNViD_1k_features"
        self.save_video_feat = save_video_feat
        ann_paths = ["1K/KoNViD_1k_videos/Konvid-1k_train_ds.json", "1K/KoNViD_1k_videos/Konvid-1k_test_ds_repaired.json"]  #  "LBVD_test_ds.json"
        q_mos_path = os.path.join(data_path, 'a_prompt/prompt_list_noTask.json')
        q_ass_path = os.path.join(data_path, 'a_prompt/prompt_list_noTask_ass.json')

        self.Q_MOS = json.load(open(q_mos_path, 'r'))
        self.Q_ASS = json.load(open(q_ass_path, 'r'))
        self.video_loader_type = video_loader_type
        self.feature_root = feature_root

        real_ann_paths = []
        for ann_path in ann_paths:
            ann_path = f"{data_path}/{ann_path}"
            real_ann_paths.append(ann_path)
            assert os.path.isfile(ann_path), f"KokiDataset annotation file {ann_path} not found, you need to identify it from your folder"
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, real_ann_paths, annotation_key="annotations", **kwargs)

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=True):
        # random select a question

        if ann['ann_type'] == 'score':
            question = random.choice(self.Q_MOS)
            answer = ann["score"]
        elif ann['ann_type'] == 'class':
            question = random.choice(self.Q_ASS)
            answer = ann['class']

        instruction = self.prompter(question, with_image=True, first_message=first_message)
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        return dict(instruction=instruction, answer=answer)

    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        if self.save_video_feat:
            feature_path = os.path.join(self.feature_root, ann["image_id"])
            image_list = []
            for img_feat_file in os.listdir(feature_path):
                img_feat = torch.load(os.path.join(feature_path, img_feat_file)).unsqueeze(0)
                image_list.append(img_feat)
            images = torch.cat(image_list, 0).mean(0)
            return images
        else:
            video_path = os.path.join(self.vis_root, ann["image_id"] + ".mp4")
            video_data = load_and_transform_video(video_path,
                                     video_decode_backend=self.video_loader_type, clip_start_sec=0.0,
                                     clip_end_sec=None, num_frames=8, sample_fps=1)
            image_list = []
            for image in video_data:
                # save_debug_image(image_path, data_debug_path, data_debug_counter, get_rank(), img_idx=0)
                # image = Image.open(image_path).convert("RGB")

                image = self.vis_processor(image)  #(720, 1280, 3)
                try:
                    image = image['pixel_values'][0]
                    image_list.append(image)
                except:
                    continue
            images = np.stack(image_list, 0)
            return images

