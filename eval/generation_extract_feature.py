#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import csv
import sys
# from PIL import Image

import torch
import deepspeed

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0, to_device
from utils.model.modeling_dsvl_extract_feature import create_dsvl_model_and_transforms_extract_feature as create_model_and_transforms
# import utils.data.DST as DST
from utils.data.lsvq_dataset import load_and_transform_video

# from typing import Iterable
from transformers import AutoTokenizer, set_seed
import json
# import collections
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import pandas as pd

def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


# usage
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    # parser.add_argument('--question-format',
    #                     type=str,
    #                     default="prompt_qa",
    #                     choices=['prompt_qa', 'prompt_choice'],
    #                     help='question-format')
    # parser.add_argument('--question',
    #                     type=str,
    #                     default="please describe the image",
    #                     help='question-format')
    # parser.add_argument(
    #     "--lm_model_name_or_path",
    #     type=str,
    #     help=
    #     "Path to pretrained model or model identifier from huggingface.co/models.",
    #     required=True,
    # )
    parser.add_argument("--vision_model_name_or_path", default="openai/clip-vit-large-patch14", type=str)
    # parser.add_argument(
    #     "--pretrained_path",
    #     default=None,
    #     type=str,
    #     help="path to pretrained model",
    # )
    # parser.add_argument(
    #     "--image_token_length",
    #     type=int,
    #     default=256,
    #     help="The maximum sequence length.",
    # )
    # parser.add_argument(
    #     "--max_seq_len",
    #     type=int,
    #     default=4096,
    #     help="The maximum sequence length.",
    # )
    # parser.add_argument(
    #     "--checkpoint_path",
    #     default=None,
    #     type=str,
    #     help="path to pretrained model",
    # )
    # parser.add_argument('--checkpoint_names',
    #                     nargs='*',
    #                     default=['runing_check_stage2_v3_epoch10', ],
    #                     help='Path to the training dataset. Accepted format:'
    #                          '1) a single data path, 2) multiple datasets in the'
    #                          'form: dataset1-path dataset2-path ...')
    # parser.add_argument(
    #     "--model_name",
    #     default="dsvl",
    #     type=str,
    #     choices=["dsvl", "toy"],
    #     help="path to pretrained model",
    # )
    # parser.add_argument(
    #     "--enable_mmca_attention",
    #     action='store_true',
    #     help="enable the new proposed attn, which is similar to cross attention",
    # )
    parser.add_argument(
        "--vis_proj",
        type=str,
        default='baseline',
        help="baseline, vit, or perceiver",
    )
    parser.add_argument(
        "--eval_data",
        default="dsvl",
        type=str,
        help="path to eval data",
    )
    parser.add_argument(
        "--eval_root",
        default="data/video_score_data/LSVQ/LSVQ_videos"
    )

    # parser.add_argument(
    #     "--output_filename",
    #     default="results",
    #     type=str,
    #     help="path to eval data",
    # )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--video_loader_type",
        type=str,
        default='opencv',
        help="opencv, decord, pytorchvideo"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save feature"
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    with open(os.path.join(args.eval_root, f'{args.eval_data}.json'), 'r') as file:
        data = json.load(file)

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path,
                                            #   fast_tokenizer=True)
    # tokenizer.padding_side = 'right'
    model, image_processor = create_model_and_transforms(
        args=args,
    )


    feature_save_path = Path(args.save_path)
    error_case = []


    # model = model.cuda().half()
    model = model.eval()
    # model.projection = model.projection.to('cuda')
    model.vis_encoder = model.vis_encoder.to('cuda')
    model = model.half()
    print_rank_0(model)
    for eval_idx, ann in tqdm(enumerate(data['annotations'])):
        if ann['ann_type'] == 'class':
            continue
        video_name = ann["image_id"]
        if ".mp4" not in video_name:
        
            video_path = os.path.join(args.eval_root, video_name + ".mp4")
        else:
            video_path = os.path.join(args.eval_root, video_name)
        try:
            video_data = load_and_transform_video(video_path,
                                                video_decode_backend=args.video_loader_type, clip_start_sec=0.0,
                                                clip_end_sec=None, num_frames=4)
        except:
            print(f'{args.eval_data}-------------------------------------{video_path}')
            error_case.append(video_path)
            continue
        print(f'{args.eval_data}-------------------------------------{video_path}')

        image_list = []
        for image in video_data:
            image = image_processor.preprocess(image)  # (720, 1280, 3)  #
            try:
                image = image['pixel_values'][0]
                image_list.append(image)
            except:
                continue
        if len(image_list) > 0:

            image_tensor = torch.from_numpy(np.stack(image_list, 0)).unsqueeze(0).cuda().half() # cat all images
        else:

            raise ValueError("No image provided. Did not fix this in the modeling side yet.")


        img_feature = model.generate(image_tensor)
        # img_feature = model.generate(image_tensor, input_ids, generation_length=256)


        img_feature = img_feature.squeeze()  ## out: [8, 257, 1024]
        
        save_path_prefix = feature_save_path / str(video_name)
        if not os.path.exists(save_path_prefix):
            os.makedirs(save_path_prefix)    
        for frame_id in range(img_feature.size()[0]):
            frame_name = str(frame_id) + ".pt"
            torch.save(img_feature[frame_id,:,:],save_path_prefix / frame_name)
        # print(video_name + " "+ frame_name +" saved")
        
        tmp = pd.DataFrame({'error case': error_case})
        tmp.to_csv(f'{args.save_path}/error_case.csv')    



if __name__ == "__main__":
    main()
