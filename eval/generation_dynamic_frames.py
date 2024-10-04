#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import csv
import sys
from PIL import Image

import torch
import deepspeed

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0, to_device
from utils.model import create_dsvl_model_and_transforms as create_model_and_transforms
import utils.data.DST as DST
from utils.data.lsvq_dataset import load_and_transform_video

from typing import Iterable
from transformers import AutoTokenizer, set_seed
import json
import collections
import numpy as np
import random
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


# usage
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--question-format',
                        type=str,
                        default="prompt_qa",
                        choices=['prompt_qa', 'prompt_choice'],
                        help='question-format')
    parser.add_argument('--question',
                        type=str,
                        default="please describe the image",
                        help='question-format')
    parser.add_argument(
        "--lm_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--vision_model_name_or_path", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--image_token_length",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument('--checkpoint_names',
                        nargs='*',
                        default=['runing_check_stage2_v3_epoch10', ],
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')
    parser.add_argument(
        "--model_name",
        default="dsvl",
        type=str,
        choices=["dsvl", "toy"],
        help="path to pretrained model",
    )
    parser.add_argument(
        "--enable_mmca_attention",
        action='store_true',
        help="enable the new proposed attn, which is similar to cross attention",
    )
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
        default="/data1/zhangyu/own_data/VQA/youtube_ugc/h264/"
    )
    parser.add_argument(
        "--eval_feat",
        # default="/data1/zhangyu/own_data/VQA/LSVQ/LSVQ_feature_1fps"
        default="/data1/zhangyu/own_data/VQA/youtube_ugc/youtube_ugc_feature_1fps"
    )

    parser.add_argument(
        "--output_filename",
        default="results",
        type=str,
        help="path to eval data",
    )
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
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    with open(os.path.join(args.eval_root, f'{args.eval_data}.json'), 'r') as file:
        data = json.load(file)
    Q_MOS = json.load(open(os.path.join(args.eval_root, '../../a_prompt/prompt_list_noTask.json'), 'r'))
    Q_ASS = json.load(open(os.path.join(args.eval_root, '../../a_prompt/prompt_list_noTask_ass.json'), 'r'))

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.padding_side = 'right'
    model, image_processor, tokenizer = create_model_and_transforms(
        text_tokenizer=tokenizer,
        ds_config=None,
        args=args,
    )

    for ck_name in args.checkpoint_names:
        print(ck_name)
        get_results = collections.defaultdict(list)
        model_type = args.checkpoint_path.split('/')[-1]
        ck_path = os.path.join(args.checkpoint_path, ck_name)
        print(ck_path)
        if ck_path is not None:
            model.load_state_dict(torch.load(os.path.join(ck_path, 'pytorch_model.bin'), map_location='cpu'),
                                  strict=False)  # Z3 wouldn't save pos embeddings (vis and rope)
        else:
            Warning("No checkpoint loaded so you cannot genereate meaningful results")
        # model = model.cuda().half()
        model = model.eval()
        model.projection = model.projection.to('cuda')
        model.vis_encoder = model.vis_encoder.to('cuda')
        model = model.bfloat16()
        print_rank_0(model)
        for eval_idx, ann in tqdm(enumerate(data['annotations'])):
            video_name = ann["image_id"]
            if ann['ann_type'] == 'class':
                continue
            video_path = os.path.join(args.eval_root, video_name + ".mp4")
            print(f'{args.eval_data}-------------------------------------{video_path}')
            feature_path = os.path.join(args.eval_feat, video_name)
            if not os.path.exists(feature_path):
                # print(feature_path)
                feature_path += ".mp4"

            image_list = []
            for img_feat_file in os.listdir(feature_path):
                img_feat = torch.load(os.path.join(feature_path, img_feat_file)).unsqueeze(0)
                image_list.append(img_feat)


            # video_data = load_and_transform_video(video_path,
            #                                       video_decode_backend=args.video_loader_type, clip_start_sec=0.0,
            #                                       clip_end_sec=None, num_frames=8)
            # print(f'{args.eval_data}-------------------------------------{video_path}')
            system_instruct = []
            TEMPLATE = DST.Prompter()  # get template
            image_token_dict = DST.get_image_num_map(tokenizer)
            image_num = 0
            # image_list = []
            # for image in video_data:
            #     image = image_processor.preprocess(image)  # (720, 1280, 3)  #
            #     try:
            #         image = image['pixel_values'][0]
            #         image_list.append(image)
            #     except:
            #         continue
            if len(image_list) > 0:
                with_image = True
                image_num = 1
                image_tensor = torch.cat(image_list, 0).mean(0).unsqueeze(0).bfloat16() # cat all images
            else:
                with_image = False
                image_num = 0
                raise ValueError("No image provided. Did not fix this in the modeling side yet.")

            if ann['ann_type'] == 'score':
                question = random.choice(Q_MOS)
                answer = ann["score"]
            elif ann['ann_type'] == 'class':
                question = random.choice(Q_ASS)
                answer = ann['class']
            full_prompt = TEMPLATE(question, with_image=with_image, first_message=True,
                                   num_images=image_num)
            full_prompt_ids = tokenizer(full_prompt).input_ids  # remove bos token
            if with_image:
                image_number = 1  # noqa
                index = full_prompt_ids.index(image_token_dict[DST.DEFAULT_HUMAN_IMAGE_PRETOKEN])
                full_prompt_ids[index] = image_token_dict[DST.image_mapping_dict[str(image_number)]]
            full_prompt_ids = DST.flatten(full_prompt_ids)
            input_ids = torch.as_tensor(
                [system_instruct + full_prompt_ids]).cuda()  # entire input as system instruction for simplicity
            print('\n', round, question)

            generate_output = model.generate(image_tensor, input_ids,
                                             generation_length=256)
            # generation_kwargs={ 'num_beams':2,'num_return_sequences':1,'top_p':1,'do_sample':True, 'temperature':1}
            print('ds-vqa-->', generate_output[1], answer)
            get_results[video_name].append([question, generate_output[1], answer])
            extend_ids = generate_output[0].cpu().tolist()[0]
            while extend_ids[-1] == tokenizer.pad_token_id:
                extend_ids.pop()
            while extend_ids[0] == tokenizer.bos_token_id:
                # llama-2 generates bos token at the beginning
                extend_ids.pop(0)
            system_instruct = system_instruct + full_prompt_ids + extend_ids  # entire input as system instruction for simplicity
            system_instruct = system_instruct + [tokenizer.eos_token_id]  # add eos token
        with open(f'{args.output_filename}/pred_{model_type}_{ck_name}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['test_name', 'question', 'pred', 'answer'])
            for test_name, responses in get_results.items():
                for response in responses:
                    print(response)
                    writer.writerow([test_name] + response)
        print_rank_0("infer done !!")


if __name__ == "__main__":
    main()
