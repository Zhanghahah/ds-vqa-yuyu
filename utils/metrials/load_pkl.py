"""
load pickle for feature testing
"""
import argparse
import pickle
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "extract features from pkl file lists")

    parser.add_argument(
        "--embed_path_prefix",
        type=str,
        default='/data1/zhangyu/own_data/VQA/KVQ/embeding/train/',
        help="LLM embedding path prefix",
    )

    args = parser.parse_args()
    return args

def load_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

if __name__ == '__main__':
    args = parse_args()
    embed_path_prefix = args.embed_path_prefix
    for feat_file in tqdm(os.listdir(embed_path_prefix)):
        embedding_path = os.path.join(embed_path_prefix, feat_file)
        ret = load_pickle(embedding_path)
    print("done!")

