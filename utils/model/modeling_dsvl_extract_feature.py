import torch

from transformers import AutoConfig
from transformers import CLIPVisionModel, CLIPImageProcessor 

from .third_party_model.qwen_clip.qwen_clip import VisionTransformer
from torch import nn

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from einops import rearrange


def task_identifier(img):
    do_extract_feat = True
    if len(img.shape) == 5:
        bs, fn, _, _, _ = img.shape
        img = rearrange(img, 'b t c h w -> (b t) c h w')
    elif len(img.shape) == 4:
        bs, _, _, _ = img.shape

    elif len(img.shape) == 3:
        bs, seqlen, _ = img.shape
        do_extract_feat = False
    return do_extract_feat, img, bs


def create_dsvl_model_and_transforms_extract_feature(
        args=None):
    assert args.vision_model_name_or_path is not None
        # text_tokenizer=None,
        # ds_config=None,

    if 'qwen' in args.vision_model_name_or_path.lower():
        # use a fake config for consistent
        vis_config = AutoConfig.from_pretrained(args.vision_model_name_or_path)
        vis_config = vis_config.vision_config
        vis_encoder = VisionTransformer(
            image_size=448,
            patch_size=vis_config.patch_size, #
            width=vis_config.hidden_size,  # 1664
            layers=vis_config.num_hidden_layers,
            heads=vis_config.num_attention_heads,
            mlp_size=vis_config.intermediate_size,
            output_dim=4096,
        ) 
        vis_encoder.load_state_dict(torch.load(os.path.join(args.vision_model_name_or_path, 'pytorch_model.bin'), map_location='cpu'), strict=True)
        vis_config.hidden_size = 4096  # we need to change the hidden size to 4096
    elif 'clip' in args.vision_model_name_or_path.lower():
        vis_encoder = CLIPVisionModel.from_pretrained(args.vision_model_name_or_path)

        vis_config = vis_encoder.config
    else:
        raise ValueError("We currently only support qwen's modifed clip and other clip models")

    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model_name_or_path)
    

    model = DeepSpeedViLModel(vis_encoder, vis_config=vis_config, args=args)


    return model, image_processor


class DeepSpeedViLModel(nn.Module):


    def __init__(self, vis_encoder,
                    vis_config=None, 
                    args=None):
        super().__init__()
        self.vis_encoder = vis_encoder
        self.args = args

     
        
    def _init_weight(self):
        self.vis_encoder.requires_grad_(False)  
    
    @torch.no_grad()
    def generate(self, img
            ):  
        do_extract_feat, img, bs = task_identifier(img)
        if do_extract_feat:
            img_feature = self.vis_encoder(img)
            if not isinstance(img_feature, torch.Tensor):
                img_feature = img_feature.last_hidden_state
                if img_feature.shape[0] != bs:
                    img_feature = rearrange(img_feature, '(bs fn) seq hid -> bs fn seq hid', bs=bs)
                    return img_feature
                    img_feature = torch.mean(img_feature, 1)  # this should output (bs, seqlen, dim)
                print("error!!!!")
                return img_feature.unsqueeze()

