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
from tqdm import tqdm


def load_and_transform_video(video_path,
                             video_decode_backend='opencv',
                             clip_start_sec=0.0,
                             clip_end_sec=None,
                             num_frames=4,
                             sample_fps=1,
                             ):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    elif video_decode_backend == 'decord':
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()
        sample_gap = round(video_fps / sample_fps)
        sample_frame_idx = [i for i in range(0, total_frame_num, sample_gap)]
        video_data = vr.get_batch(sample_frame_idx).asnumpy()  # (f, h, w, 3)
        frame_time = [x / video_fps for x in sample_frame_idx]

        # decord.bridge.set_bridge('torch')
        # decord_vr = VideoReader(video_path, ctx=cpu(0))
        # duration = len(decord_vr)
        # frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        # video_data = decord_vr.get_batch(frame_id_list)
        # video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)  ?

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)

        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        if duration == 0:
            frame_id_list = np.array([0])
        else:
            frame_id_list = np.linspace(0, duration - 1, duration, dtype=int)

        video_data = []
        print_rank_0(f"Processing video {video_path}, the num of frame_id_list: {len(frame_id_list)}")
        for frame_idx in tqdm(frame_id_list):
            cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cv2_vr.read()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                if frame is None:
                    continue
                # try:
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # except:
            #     while frame is None:
            #         random_frame_idx = np.random.choice((duration - 1) // 2)
            #         cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
            #         _, frame = cv2_vr.read()
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('./data_debug_vis/debug.jpg', frame)
            # frame = Image.open('./data_debug_vis/debug.jpg').convert("RGB")

            video_data.append(frame)
        cv2_vr.release()
        assert len(video_data) != 0, "We need at least one frame for feature extraction."
        # if len(video_data) == 0:
        #     """we need to give a dummy video frame for negative smaple learning"""

    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_data


class LSVQAlignDataset(VQADataset):
    def __init__(self, data_path,
                 data_debug_path,
                 per_sample_image,
                 video_loader_type,
                 tokenizer,
                 vis_processor,
                 add_eos=True, ignore_instruction=True, save_video_feat=True, **kwargs):
        vis_root = f"{data_path}/LSVQ/LSVQ"
        feature_root = "/data1/zhangyu/own_data/VQA/LSVQ/LSVQ_feature_8B_bf16"
        motion_root = "/data1/zhangyu/own_data/VQA/LSVQ/LSVQ_SlowFast2_feature"
        assert os.path.isdir(
            vis_root), f"LSVQAlignDataset image directory {vis_root} not found, you need to check the image path"

        ann_paths = [
            "LSVQ/LSVQ/a_split_metadata/LSVQ_whole_train_ds_score.json",
            "LSVQ/LSVQ/a_split_metadata/LSVQ_whole_test_ds_score.json",
            "LSVQ/LSVQ/a_split_metadata/LSVQ_whole_train_ds_class.json",
            "LSVQ/LSVQ/a_split_metadata/LSVQ_whole_test_ds_class.json",
            "LSVQ/LSVQ/a_split_metadata/LSVQ_whole_test_1080p_ds_class.json",
            "LSVQ/LSVQ/a_split_metadata/LSVQ_whole_test_1080p_ds.json"]
        q_mos_path = os.path.join(data_path, 'a_prompt/prompt_list_noTask.json')
        q_ass_path = os.path.join(data_path, 'a_prompt/prompt_list_noTask_ass.json')

        self.Q_MOS = json.load(open(q_mos_path, 'r'))
        self.Q_ASS = json.load(open(q_ass_path, 'r'))
        self.video_loader_type = video_loader_type
        self.feature_root = feature_root
        self.save_video_feat = save_video_feat

        self.motion_root = motion_root
        self.ext_motion_token = True

        real_ann_paths = []
        for ann_path in ann_paths:
            ann_path = f"{data_path}/{ann_path}"
            real_ann_paths.append(ann_path)
            assert os.path.isfile(
                ann_path), f"LSVQAlignDataset annotation file {ann_path} not found, you need to identify it from your folder"
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, real_ann_paths, annotation_key="annotations", **kwargs)

    def motion_feature_encode(self, slow_mo_feat_name, fast_mo_feat_name):

        slow_mo_feat = torch.from_numpy(np.load(slow_mo_feat_name)).squeeze().cuda()
        fast_mo_feat = torch.from_numpy(np.load(fast_mo_feat_name)).squeeze().cuda()
        feat_3D = torch.cat([slow_mo_feat, fast_mo_feat]).unsqueeze(0)
        return feat_3D

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
            if not os.path.exists(feature_path):
                feature_path = os.path.join(self.feature_root, ann["image_id"] + ".mp4")
            image_list = []
            motion_feat_list = []
            # find the max motion token
            max_motion_idx = sorted(os.listdir(os.path.join(self.motion_root, ann["image_id"])))[-1].split('_')[1]
            feat_path_list = sorted(os.listdir(feature_path))
            fs_count = len(feat_path_list)
            if fs_count < 8:
                feat_path_list += [feat_path_list[-1]] * (8 - fs_count)
            elif fs_count > 8:
                feat_path_list = feat_path_list[:8]
            for img_feat_file in feat_path_list:
                img_index = img_feat_file.split('.')[0]
                # if img_index > max_motion_idx:c
                #     continue
                if self.ext_motion_token:
                    # [1,2304]
                    slow_mo_feat = os.path.join(self.motion_root, ann["image_id"],
                                                f"feature_{img_index}_slow_feature.npy")
                    fast_mo_feat = os.path.join(self.motion_root, ann["image_id"],
                                                f"feature_{img_index}_fast_feature.npy")

                    if not os.path.exists(slow_mo_feat) or not os.path.exists(fast_mo_feat):
                        slow_mo_feat = os.path.join(self.motion_root, ann["image_id"],
                                                    f"feature_{max_motion_idx}_slow_feature.npy")
                        fast_mo_feat = os.path.join(self.motion_root, ann["image_id"],
                                                    f"feature_{max_motion_idx}_fast_feature.npy")
                    motion_feat_per_img = self.motion_feature_encode(slow_mo_feat, fast_mo_feat)
                    motion_feat_list.append(motion_feat_per_img)
                img_feat = torch.load(os.path.join(feature_path, img_feat_file)).unsqueeze(0)
                image_list.append(img_feat)

            if self.ext_motion_token:
                # if we add motion token for each image, we return two list: image_list and motion_list
                # img_mo_feats_list = [torch.cat([img_feat, mo_feat], dim=0)
                #                      for img_feat, mo_feat in zip(image_list, motion_feat_list)]
                return [torch.cat(image_list, 0), torch.cat(motion_feat_list, 0)]
            else:
                # if we do not use motion token, we just return a image tensor
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

                image = self.vis_processor(image)  # (720, 1280, 3)
                try:
                    image = image['pixel_values'][0]
                    image_list.append(image)
                except:
                    continue
            images = np.stack(image_list, 0)
            return images
