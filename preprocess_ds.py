import os
import glob
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import random

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

def parse_csv(attribute_path):

    attri_data = pd.read_csv(attribute_path)
    attri_data = attri_data[["name", "mos"]]

    print(f"load data done.")
    return attri_data

def parse_excel_LBVD(attribute_path):

    attri_data = pd.read_excel(attribute_path,header= None)
    attri_data.columns = ["mos", "varience"]
    attri_data['name'] = range(1,len(attri_data)+1)

    print(f"load data done.")
    return attri_data

def parse_excel_Gaming(attribute_path):

    attri_data = pd.read_excel(attribute_path,header= None)
    attri_data.columns = ["name", "mos"]

    print(f"load data done.")
    return attri_data

def parse_excel_ugc(attribute_path):

    attri_data = pd.read_excel(attribute_path, sheet_name="MOS")
    attri_data = attri_data[["vid", "MOS full"]]
    attri_data.columns = ["name", "mos"]

    print(f"load data done.")
    return attri_data

def parse_prompts(prompt_path_prefix):
    iqa_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'prompts/vqa_question.txt'
        ), "r").readlines()

    iqa_task_desc_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'prompts/vqa_task_descriptor.txt'
        ), "r").readlines()
    
    iqa_ass_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'prompts/vqa_question_ass.txt'
        ), "r").readlines()
    
    iqa_ass_limitation_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'prompts/vqa_limitation_ass.txt'
        ), "r").readlines()

    iqa_limitation_prompts_100 = open(
        os.path.join(
            prompt_path_prefix,
            'prompts/vqa_limitation_100.txt'
        ), "r").readlines()
        
    
    return iqa_prompts, iqa_task_desc_prompts, iqa_limitation_prompts_100, iqa_ass_prompts, iqa_ass_limitation_prompts



def score2class(score, score_item):
    if score > score_item['66%']:
        result = 'good'
    elif score <= score_item['66%'] and score > score_item['33%']:
        result = 'fair'
    elif score <= score_item['33%'] and score > 0:
        result = 'poor'
    else:
        print('error score:')
        print(score)
        exit()
    return result

def score2floor(score):
    result = format(score, '.3f')
    return result


def pd2dict(iqa_score, config):
    print(iqa_score.loc[0:1])
    data_list = iqa_score.to_dict('records')
    print(data_list[0:1])
   
    data_dict = {line['name']: [line['task_descript'] + ' ' + line['question'] + ' ' + line['limitation'], line['mos']] for line in data_list}
    if config.have_ass:
        data_dict_ass = {line['name']: [line['task_descript'] + ' ' + line['ass_question'] + ' '+ line['ass_limitation'], score2class(line['mos'])] for line in data_list}
        return data_dict, data_dict_ass
        


def pd2list(iqa_score, config, score_item):      

    data_list = []
    for i in range(len(iqa_score)):
        item_dict = {}
        item_dict['image_id'] = str(iqa_score.loc[i,'name'])
        item_dict['ann_type'] = 'score'
        item_dict['score'] = score2floor(iqa_score.loc[i,'mos'])
        item_dict_ass = {}
        item_dict_ass['image_id'] = str(iqa_score.loc[i,'name'])
        item_dict_ass['ann_type'] = 'class'
        item_dict_ass['class'] = score2class(iqa_score.loc[i,'mos'], score_item)
        data_list.append(item_dict)
        data_list.append(item_dict_ass)
    return data_list

def save_js(data, path):
        with open(path, "wt") as f:
            json.dump(data, f)


def IQA(config):

    set_seed(123)

    if config.dataset == 'Konvid-1k' or config.dataset == 'KVQ':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_csv(config.metadata_path) # "flickr_id","mos"
        iqa_score.columns = ["name", "mos"]
        iqa_score["mos"] = iqa_score["mos"] * 20
        print(f"load data done.")

    elif config.dataset == 'LSVQ' or config.dataset == 'LIVE-VQC' or config.dataset == 'LIVE-YT-Gaming':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_csv(config.metadata_path) 
        iqa_score = iqa_score[["name", "mos"]]
        print(f"load data done.")
    
    elif config.dataset == 'koniq10k':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_csv(config.metadata_path) 
        iqa_score = iqa_score[["image_name", "MOS_zscore"]]
        iqa_score.columns = ["name", "mos"]
        print(f"load data done.")

    elif config.dataset == 'LBVD':
        print('The current model is ' + config.dataset)
        iqa_score = parse_excel_LBVD(config.metadata_path)
        iqa_score["mos"] = iqa_score["mos"] * 20
        print("max")
        print(max(iqa_score["mos"]))
        print("min")
        print(min(iqa_score["mos"]))
        print(f"load data done.")   
    
    elif config.dataset == 'LGHVQ':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_csv(config.metadata_path, sep=';')
        iqa_score = iqa_score[["video_path", "mos"]]
        iqa_score.columns = ["name", "mos"]
        print(f"load data done.")


    elif config.dataset == 'YT-ugc':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_csv(config.metadata_path) 
        iqa_score["mos"] = iqa_score["mos"] * 20
        print(f"load data done.")

    print(iqa_score.loc[0:1])

    iqa_prompts, iqa_task_desc_prompts, iqa_limitation_prompts_100, iqa_ass_prompts, iqa_ass_limitation_prompts = parse_prompts(os.getcwd())

    # out_js_name = os.getcwd() /  Path(config.dataset)
    in_js_name = config.metadata_path.rsplit('.',1)[0].split('/')[-1]
    out_path = os.getcwd() /  Path('result/ds_five')


    iqa_score.insert(iqa_score.shape[1], 'question', None)
    iqa_score.insert(iqa_score.shape[1], 'task_descript', None)

    # out_js = {}
    # for i in tqdm(range(len(iqa_score))):
    #     iqa_score.loc[i, 'question'] = np.random.choice(iqa_prompts).strip()
    #     iqa_score.loc[i, 'task_descript'] = np.random.choice(iqa_task_desc_prompts).strip()
    #     iqa_score.loc[i, 'limitation'] = np.random.choice(iqa_limitation_prompts_100).strip()
    # if config.have_ass:
    #     for i in tqdm(range(len(iqa_score))):
    #         iqa_score.loc[i, 'ass_question'] = np.random.choice(iqa_ass_prompts).strip()
    #         iqa_score.loc[i, 'ass_limitation'] = np.random.choice(iqa_ass_limitation_prompts).strip()




    ## prompt for ds; with task_desc: prompt_list/prompt_list_ass; without task_desc: 
    prompt_list = []
    prompt_list_ass = []
    for i in range(20):
        if i < 10:
            # prompt_list.append(iqa_task_desc_prompts[i].strip() + ' ' + iqa_prompts[i].strip() + ' ' + iqa_limitation_prompts_100[i].strip())
            prompt_list.append(iqa_prompts[i].strip() + ' ' + iqa_limitation_prompts_100[i].strip())
        else:
            # prompt_list.append(iqa_task_desc_prompts[i%10].strip() + ' ' + iqa_prompts[i].strip() + ' ' + iqa_limitation_prompts_100[(i+1)%10].strip())
            prompt_list.append(iqa_prompts[i].strip() + ' ' + iqa_limitation_prompts_100[(i+1)%10].strip())
    for i in range(20):
        if i < 10:
            # prompt_list_ass.append(iqa_task_desc_prompts[i].strip() + ' ' + iqa_ass_prompts[i].strip() + ' ' + iqa_ass_limitation_prompts[i].strip())
            prompt_list_ass.append(iqa_ass_prompts[i].strip() + ' ' + iqa_ass_limitation_prompts[i].strip())
        else:
            # prompt_list_ass.append(iqa_task_desc_prompts[i%10].strip() + ' ' + iqa_ass_prompts[(i+1)%10].strip() + ' ' + iqa_ass_limitation_prompts[(i+2)%10].strip()) 
            prompt_list_ass.append(iqa_ass_prompts[(i+1)%10].strip() + ' ' + iqa_ass_limitation_prompts[(i+2)%10].strip()) 



    ############### output one file ###############
    score_item = iqa_score['mos'].describe(percentiles=[.33,0.66])
    print("score_item")
    print(score_item)

    data_list = pd2list(iqa_score, config, score_item)
    data_dict = {'annotations': data_list}
    save_js(data_dict,str(out_path / config.dataset) + "_ds.json")   


    ############### save prompt list ###############
    # out_js_name = in_js_name + "_ds.json"   
    # out_js_path = out_path / out_js_name
    # print('outpath: '+ str(out_js_path))
    # save_js(prompt_list, out_path / 'prompt_list_noTask.json')
    # save_js(prompt_list_ass, out_path / 'prompt_list_noTask_ass.json')
               


    ############### output five file ###############
    # score_item = iqa_score['mos'].describe(percentiles=[.33,0.66])
    # print("score_item")
    # print(score_item)
    # score_shuffled = iqa_score.sample(frac=1, random_state=1).reset_index(drop=True)
    # partitions = np.array_split(score_shuffled, 5)
    # for i, partition in enumerate(partitions):
    #     data_list = pd2list(partition.reset_index(drop=True), config, score_item)
    #     data_dict = {'annotations': data_list}
    #     save_js(data_dict,str(out_path / config.dataset) + "_" + str(i) + "_ds.json")

  

    ############### output test & train ###############
    # iqa_score_train = iqa_score.sample(n = int(len(iqa_score)*0.8))
    # iqa_score_test = iqa_score.drop(iqa_score_train.index).reset_index(drop=True)
    # iqa_score_train = iqa_score_train.reset_index(drop=True)

    # data_list_train = pd2list(iqa_score_train, config)
    # data_dict_train = {'annotations': data_list_train}
    # save_js(data_dict_train,str(out_path / config.dataset) + "_train_ds.json")

    # data_list_test = pd2list(iqa_score_test, config)
    # data_dict_test = {'annotations': data_list_test}
    # save_js(data_dict_test,str(out_path / config.dataset) + "_test_ds.json")


   ############### config.have_ass ###############
    # if config.have_ass:   
        # out_js_ass_path = str(out_js_name) + "_ass_ds.json"
        # save_js(data_dict_ass,out_js_ass_path)

    # if 'is_test' in iqa_score:
    #     print('output training/testing set metadata')
    #     iqa_score_train = iqa_score.query('is_test == False')       
    #     iqa_score_test = iqa_score.query('is_test == True')
               
    # else:
        # iqa_score_train = iqa_score.sample(n = int(len(iqa_score)*0.8))
        # iqa_score_test = iqa_score.drop(iqa_score_train.index)    

    # data_dict_train = pd2dict(iqa_score_train, config)
    # out_js_path = str(out_js_name) + "_train.json"
    # save_js(data_dict_train,out_js_path)

    # data_dict_test = pd2dict(iqa_score_test, config)  
    # out_js_path = str(out_js_name) + "_test.json"
    # save_js(data_dict_test,out_js_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset', type=str, help='Konvid-1k or LSVQ or LBVD or LIVE-VQC or LIVE-YT-Gaming or YT-ugc or KVQ or LGHVQ',default='YT-ugc')
    parser.add_argument('--metadata_path', type=str, default= './data/origin_data/youtube_ugc_whole.csv')
    parser.add_argument('--have_ass', action='store_true', help="save the attentions and hidden state or not.") # not used

    config = parser.parse_args()

    IQA(config)