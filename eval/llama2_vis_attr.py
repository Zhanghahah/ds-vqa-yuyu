import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    ProductBaselines,
)


def llama2_token_vis_attr(model,
                          tokenizer,
                          output_response,):
    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)
    print("attr to the output sequence:", attr_res.seq_attr.shape)  # shape(n_input_token)
    print("attr to the output tokens:", attr_res.token_attr.shape)  # shape(n_output_token, n_input_token)
