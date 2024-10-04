import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import os
import json
import re
import pandas as pd
import argparse


## y_output = np.zeros([len(ouput_length)])
## y_label = np.zeros([len(ouput_length)])
class postprocess():
    def __init__(self) -> None:
        self.count = {}
        self.error_score = {}
        # self.output_id = None

    def extract_first_digit(self, text_raw):
        # pattern = r"\d+"  # 匹配一个或多个数字字符
        # pattern1 = r'\d+\.\d+'
        # pattern2 = r'\d+\.'
        # match = re.search(pattern, text)
        # if match:
        #     return int(match.group())  # 返回匹配到的第一个数字
        # else:
        #     return None
        # num_patterns =  [r'\d+\.\d+', r'\d+\.', r"\d+"]
        # num_patterns = re.compile("\d+[.\d]*") #所有整数或小数

        if text_raw['output_id'] not in self.count.keys():
            self.count[text_raw['output_id']] = 0
            self.error_score[text_raw['output_id']] = []
        text = repr(text_raw['score'])

        if re.search(r"\d+", text) == None or float(re.search(r"\d+", text).group()) == 0:  # 如果没有数字或者数字之和为零
            self.error_case(text_raw, 'this text is 0')
            return ('60')
        elif text.count('.') != 1:
            self.error_case(text_raw, 'this text have not single .')
            # self.count+=1
            # print(self.count)
            # print('this text have not single .')
            score = re.sub('\D', '', text)  # 去除所有非数字
            score = score.lstrip('0')[:2] + '.' + score.lstrip('0')[3:]  # 输出小数
            return score
        else:
            score = re.search(r"\d+\.?\d*", text)
            score = score.group()
            if float(score) < 100 and float(score) > 10:
                return score
            else:
                score = re.sub('\D', '', text)  # 去除所有非数字
                score = score.lstrip('0')[:2] + '.' + score.lstrip('0')[3:]  # 输出小数
                if float(score) < 100 and float(score) > 10:
                    self.error_case(text_raw, 'this text is ood')
                    return score
                else:
                    self.error_case(text_raw, 'this text is very ood:')
                    return '60'

    def error_case(self, text_raw, error_str):
        self.count[text_raw['output_id']] += 1
        self.error_score[text_raw['output_id']].append(text_raw['score'])
        # print(text_raw['output_id'] + '  num: ' + str(self.count[text_raw['output_id']]))
        # print(error_str)
        # print(text_raw['score'])


def save_js(data, path):
    with open(path, "wt") as f:
        json.dump(data, f)


def essemble(pred_folder, pred_file_list):
    process = postprocess()
    # pred = process.extract_first_digit(pred)
    score_dict_list = []
    for i in range(len(pred_file_list)):
        df = pd.read_csv(pred_folder + pred_file_list[i])
        score = df.query('answer != ["poor", "fair", "good"]')
        # class = df.drop(score.index)
        score_dict = {key: {'score': value, 'output_id': pred_file_list[i]} for key, value in
                      zip(list(score['test_name'].values),
                          list(score['pred'].values))}  # video_name: [video_score, output_id]
        score_dict_list.append(score_dict)
    score_dict_essemble = {}
    for key in score_dict_list[0].keys():
        score_dict_essemble[key] = np.mean([eval(process.extract_first_digit(dict[key])) for dict in score_dict_list])


def process(pred_file):
    process = postprocess()
    df = pd.read_csv(pred_file)
    score = df.query('answer != ["poor", "fair", "good"]')
    output_id = pred_file.rsplit('.', 1)[0].split('/')[-1]
    # print('output_id')
    score_dict = {key: {'score': value, 'output_id': output_id} for key, value in
                  zip(list(score['test_name'].values), list(score['pred'].values))}

    score_dict_processed = {k: eval(process.extract_first_digit(v)) for k, v in score_dict.items()}
    # score_dict_processed = {}
    # for k, v in score_dict.items():
    #     score_dict_processed[k] = eval(process.extract_first_digit(v))

    save_js(process.count, 'count_one.json')
    save_js(process.error_score, 'error_score_one.json')
    return score_dict_processed


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic - y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE


def performance_no_fit(y_label, y_output):
    PLCC = stats.pearsonr(y_output, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_label - y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE


def pred_trans(pred):
    success_flag = True
    try:
        ans = float(pred)
    except:
        try:
            ans = float(pred.split('.')[0])
        except:
            success_flag = False
    return ans, success_flag


def metric_cal_loaded(preds_dict, labels_dict):
    preds, labels = [], []

    for key, pred in preds_dict.items():

        # ans, su_flag = pred_trans(pred)
        try:
            preds.append(float(pred))
            gt = labels_dict[key]
            labels.append(gt)
        except:
            print('video: ' + str(key) + ' is not used')
            continue
            # print(preds)

    PLCC, SRCC, KRCC, RMSE = performance_fit(np.array(labels), np.array(preds))
    print(f"PLCC: {PLCC}, SRCC: {SRCC}, KRCC: {KRCC}, RMSE: {RMSE}")
    return PLCC, SRCC, KRCC, RMSE


def metric_cal(pred_file, label_file, config):
    preds_dict = process(pred_file)
    # preds_df = pd.read_csv(pred_path)
    # preds_df_score = preds_df.query('answer != ["poor", "fair", "good"]')
    # preds_dict = {key: value for key, value in zip(list(preds_df_score['test_name'].values), list(preds_df_score['pred'].values))}

    # with open(pred_path, 'rt') as f:
    #     preds_dict = json.load(f)

    # try:
    #     with open(label_file, 'rt') as f:
    #         labels_dict = json.load(f)
    # except:
    labels_pd = pd.read_csv(label_file)
    labels_pd.columns = ["name", "mos"]
    if config.dataset == "Konvid-1k":
        labels_dict = {key: value * 20 for key, value in
                       zip(list(labels_pd['name'].values), list(labels_pd['mos'].values))}
    else:
        labels_dict = {key: value for key, value in zip(list(labels_pd['name'].values), list(labels_pd['mos'].values))}

    PLCC, SRCC, KRCC, RMSE = metric_cal_loaded(preds_dict, labels_dict)


if __name__ == '__main__':
    # data_prefix = "/home/zhangyu/data/1K/KoNViD_1k_images"
    # in_file = "/home/bml/storage/mnt/v-7db79275c2374/org/qihang/validation/stage_two_result/LSVQ/LSVQ_f448_mlr2e-7_2_train_S2.json"
    # label_file = "/home/bml/storage/mnt/v-7db79275c2374/org/qihang/data/LSVQ/LSVQ_image_mul/LSVQ_whole_test_limit.json"

    # in_file = "/data/zhangyu/ds-vqa-yuyu/eval/results/LSVQ_whole_test_ds_score/pred_epoch-1.csv"
    # label_file = "/data/zhangyu/own_data/VQA/LSVQ/LSVQ/LSVQ_whole_test_limit.json"

    # in_file = "/data/zhangyu/ds-vqa-yuyu/eval/results/Konvid-1k_test_ds/pred_DATE01-19_Epoch6_LR1e-3_data_lsvq_align_all_1_epoch-5.csv"
    # label_file = "/data/zhangyu/own_data/VQA/metadata_vqa/metadata/KoNViD_1k_mos.csv"

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset', type=str, help='Konvid-1k or LSVQ or LBVD or LIVE-VQC or LIVE-Gaming or YT-ugc')
    parser.add_argument('--in_file', type=str)

    config = parser.parse_args()
    meta_data_path = "/data/zhangyu/own_data/VQA/metadata_vqa/metadata_csv/"
    label_file_dict = {"Konvid-1k": "KoNViD_1k_mos.csv", "LSVQ": "LSVQ_whole.csv", "LBVD": "LBVD_whole.csv",
                       "LSVQ_1080p": "LSVQ_whole_test_1080p_.csv",
                       "LIVE-VQC": "LIVE-VQC_.csv", "LIVE-Gaming": "LIVE_Gaming.csv",
                       "YT-ugc": "youtube_ugc_whole.csv"}
    label_file = meta_data_path + label_file_dict[config.dataset]
    metric_cal(config.in_file, label_file, config)


# if __name__ == '__main__':
#     # data_prefix = "/home/zhangyu/data/1K/KoNViD_1k_images"
#     # in_file = "/home/bml/storage/mnt/v-7db79275c2374/org/qihang/validation/stage_two_result/LSVQ/LSVQ_f448_mlr2e-7_2_train_S2.json"
#     # label_file = "/home/bml/storage/mnt/v-7db79275c2374/org/qihang/data/LSVQ/LSVQ_image_mul/LSVQ_whole_test_limit.json"
#
#     # in_file = "/data/zhangyu/ds-vqa-yuyu/eval/results/LSVQ_whole_test_ds_score/pred_epoch-1.csv"
#     # label_file = "/data/zhangyu/own_data/VQA/LSVQ/LSVQ/LSVQ_whole_test_limit.json"
#
#     in_file = "/data/zhangyu/ds-vqa-yuyu/eval/results/YT-ugc_test_ds/pred_DATE01-19_Epoch6_LR1e-3_data_lsvq_align_all_1_epoch-5.csv"
#     label_file = "/data/zhangyu/own_data/VQA/metadata_vqa/metadata/youtube_ugc_whole.csv"
#
#     metric_cal(in_file, label_file)