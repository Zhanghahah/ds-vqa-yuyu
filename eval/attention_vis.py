import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


def visualize_token2token_scores(scores_mat,
                                 x_label_name='Head'):
    fig = plt.figure(figsize=(20, 20))

    for idx, scores in enumerate(scores_mat):
        if idx > 11:
            break
        scores_np = np.array(scores)
        ax = fig.add_subplot(4, 3, idx + 1)
        # append the attention weights
        im = ax.imshow(scores, cmap='viridis')

        fontdict = {'fontsize': 10}

        # ax.set_xticks(range(len(all_tokens)))
        # ax.set_yticks(range(len(all_tokens)))
        # ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        # ax.set_yticklabels(all_tokens, fontdict=fontdict)
        # ax.set_xlabel('{} {}'.format(x_label_name, idx + 1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig('./attention_vis.png')
    plt.tight_layout()
    plt.show()


def visualize_token2head_scores(scores_mat):
    fig = plt.figure(figsize=(30, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(6, 2, idx + 1)
        # append the attention weights
        im = ax.matshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 20}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(scores)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
        ax.set_xlabel('Layer {}'.format(idx + 1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
