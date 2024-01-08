"""
temper file for data valida

"""

import os
import json
from collections import defaultdict

def cc_sub_align_dataset_validation(dataset_path):
    ann_json_file = 'filter_cap.json'
    filter_ann_json_file = 'filtered_cap.json'
    image_file = 'image'
    filtered_ann = defaultdict(list)
    image_ids = set()
    ann_data = json.load(open(os.path.join(dataset_path, ann_json_file), "r"))
    for file in os.listdir(os.path.join(dataset_path, image_file)):
        image_id = file.split('.')[0]
        image_ids.add(image_id)
    for data in ann_data['annotations']:
        if data['image_id'] in image_ids:
            filtered_ann['annotations'].append(data)

    with open(os.path.join(dataset_path, filter_ann_json_file), 'w') as f:
        json.dump(filtered_ann, f)








if __name__ == '__main__':
    cc_sub_align_folder_path = '/data/zhangyu/own_data/VQA/cc_sbu_align'
    cc_sub_align_dataset_validation(cc_sub_align_folder_path)