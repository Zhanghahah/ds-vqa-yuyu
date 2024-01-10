import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import os
import json
import re
import pandas as pd


## y_output = np.zeros([len(ouput_length)])
## y_label = np.zeros([len(ouput_length)])
def extract_first_digit(text_raw):
    # pattern = r"\d+"  # 匹配一个或多个数字字符
    # pattern1 = r'\d+\.\d+'
    # pattern2 = r'\d+\.'
    # match = re.search(pattern, text)
    # if match:
    #     return int(match.group())  # 返回匹配到的第一个数字
    # else:
    #     return None
    text = repr(text_raw)
    if re.search(r"\d+", text) == None or float(re.search(r"\d+", text).group()) == 0:  # 如果没有数字或者数字之和为零
        print('this text is 0:')
        print(text_raw)
        return ('50')
    # num_patterns =  [r'\d+\.\d+', r'\d+\.', r"\d+"]
    # num_patterns = re.compile("\d+[.\d]*") #所有整数或小数
    if 'count' not in locals().keys():
        count = 0.
    if text.count('.') != 1:
        count += 1
        print(count)
        print('this text have not single .')
        print(text)
    score = re.search(r"\d+\.?\d*", text)
    score = score.group()
    if float(score) < 100 and float(score) > 10:
        return score
    else:
        tmp = re.sub('\D', '', text)  # 去除所有非数字
        score_h = tmp.lstrip('0')[:2]
        score_p = tmp.lstrip('0')[2:]
        score = '.'.join([score_h, score_p])
        if float(score) < 100 and float(score) > 10:
            print('this text can only output int:')
            print(text)
            return score
        else:
            print('this text is ood:')
            print(text)
            return '50'


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
        pred = extract_first_digit(pred)
        try:
            preds.append(float(pred))
            gt = labels_dict[key][-1]
            labels.append(gt)
        except:
            print('video: ' + key + ' is not used')
            continue
    print(preds)

    PLCC, SRCC, KRCC, RMSE = performance_fit(np.array(labels), np.array(preds))
    print(f"PLCC: {PLCC}, SRCC: {SRCC}, KRCC: {KRCC}, RMSE: {RMSE}")
    return PLCC, SRCC, KRCC, RMSE


def metric_cal(pred_path, label_path):
    preds_df = pd.read_csv(pred_path)
    preds_df_score = preds_df.query('answer != ["poor", "fair", "good"]')
    preds_dict = {key: value for key, value in
                  zip(list(preds_df_score['test_name'].values), list(preds_df_score['pred'].values))}

    # with open(pred_path, 'rt') as f:
    #     preds_dict = json.load(f)

    with open(label_path, 'rt') as f:
        labels_dict = json.load(f)

    PLCC, SRCC, KRCC, RMSE = metric_cal_loaded(preds_dict, labels_dict)


if __name__ == '__main__':
    # data_prefix = "/home/zhangyu/data/1K/KoNViD_1k_images"
    # in_file = "/home/bml/storage/mnt/v-7db79275c2374/org/qihang/validation/stage_two_result/LSVQ/LSVQ_f448_mlr2e-7_2_train_S2.json"
    # label_file = "/home/bml/storage/mnt/v-7db79275c2374/org/qihang/data/LSVQ/LSVQ_image_mul/LSVQ_whole_test_limit.json"
    in_file = "/data/zhangyu/ds-vqa-yuyu/eval/results/LSVQ_whole_test_ds_score/pred_epoch-5.csv"
    label_file = "/data/zhangyu/own_data/VQA/LSVQ/LSVQ/LSVQ_whole_test_limit.json"

    # pred_path, label_path  = os.path.join(data_prefix, in_file), \
    #                             os.path.join(data_prefix, label_file)
    metric_cal(in_file, label_file)