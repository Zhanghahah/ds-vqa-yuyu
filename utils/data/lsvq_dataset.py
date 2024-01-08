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

import numpy as np
import torch


def load_and_transform_video(video_path,
                             video_decode_backend='opencv',
                             clip_start_sec=0.0,
                             clip_end_sec=None,
                             num_frames=4
                             ):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)


        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cv2_vr.read()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print_rank_0(video_path)
                while frame is None:
                    random_frame_idx = np.random.choice((duration - 1) // 2)
                    cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
                    _, frame = cv2_vr.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('./data_debug_vis/debug.jpg', frame)
            # frame = Image.open('./data_debug_vis/debug.jpg').convert("RGB")

            video_data.append(frame)
        cv2_vr.release()
        # video_data = torch.stack(video_data, dim=1)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_data


class LSVQAlignDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, add_eos=True, ignore_instruction=True, **kwargs):
        vis_root = f"{data_path}/LSVQ/LSVQ"
        assert os.path.isdir(vis_root), f"CcSbuAlignDataset image directory {vis_root} not found, you need to download it from https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align"

        ann_paths = ["LSVQ/LSVQ/LSVQ_whole_test_ds.json"]
        q_mos_path = os.path.join(vis_root, 'prompt_list.json')
        q_ass_path = os.path.join(vis_root, 'prompt_list_ass.json')

        self.Q_MOS = json.load(open(q_mos_path, 'r'))
        self.Q_ASS = json.load(open(q_ass_path, 'r'))

        real_ann_paths = []
        for ann_path in ann_paths:
            ann_path = f"{data_path}/{ann_path}"
            real_ann_paths.append(ann_path)
            assert os.path.isfile(ann_path), f"LSVQAlignDataset annotation file {ann_path} not found, you need to identify it from your folder"
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
        video_path = os.path.join(self.vis_root, ann["image_id"] + ".mp4")
        video_data = load_and_transform_video(video_path,
                                 video_decode_backend='opencv', clip_start_sec=0.0,
                                 clip_end_sec=None, num_frames=8)
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
