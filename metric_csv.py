import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
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
    def extract_first_digit(self, text_raw):
        if text_raw['output_id'] not in self.count.keys():
            self.count[text_raw['output_id']] = 0
            self.error_score[text_raw['output_id']] = []
        text = repr(text_raw['score'])    
        # text = repr(text_raw['score'])
        # text = str(eval(text_raw['score'])[0]*100)

        if re.search(r"\d+", text) == None or float(re.search(r"\d+", text).group()) == 0:  
            self.error_case(text_raw, 'this text is 0')
            return ('60')
        elif text.count('.') == 0:
            score = re.sub('\D', '', text)  
            score = score.lstrip('0')[:2] + '.'
            return score
        elif text.count('.') != 1:
            self.error_case(text_raw, 'this text have not single .')
            score = re.sub('\D', '', text)  
            score = score.lstrip('0')[:2] + '.' + score.lstrip('0')[2:]  
            return score
        else:
            score = re.search(r"\d+\.?\d*", text)
            score = score.group()
            if float(score) < 100 and float(score) > 0:
                return float(score)
            else:
                score = re.sub('\D', '', text)  
                score = score.lstrip('0')[:2] + '.' + score.lstrip('0')[2:]  
                if float(score) < 100 and float(score) > 10:
                    self.error_case(text_raw, 'this text is ood')
                    return float(score)
                else:
                    self.error_case(text_raw, 'this text is very ood:')
                    return float(60)

    def error_case(self, text_raw, error_str):
        self.count[text_raw['output_id']] += 1
        self.error_score[text_raw['output_id']].append(text_raw['score'])


def save_js(data, path):
    with open(path, "wt") as f:
        json.dump(data, f)

def essemble(pred_folder, pred_file_list):
    process = postprocess()
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

    score = pd.read_csv(pred_file)
    output_id = pred_file.rsplit('.', 1)[0].split('/')[-1]

    score_dict = {key: {'score': value, 'output_id': output_id} for key, value in
                  zip(list(score['test_name'].values), list(score['pred'].values))}

    score_dict_processed = {str(k): process.extract_first_digit(v) for k, v in score_dict.items()}


    save_js(process.count, 'invalid_count_result/' + output_id + '_count_one.json')
    save_js(process.error_score, 'invalid_count_result/' + output_id + '_error_score_one.json')
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
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())
    R2 = r2_score(y_label, y_output)
    l1 = (np.absolute( y_output-y_label)).mean()

    return PLCC, SRCC, KRCC, RMSE, R2, l1


def performance_no_fit(y_label, y_output):
    PLCC = stats.pearsonr(y_output, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt((( y_output-y_label) ** 2).mean())
    R2 = r2_score(y_label, y_output)
    l1 = (np.absolute( y_output-y_label)).mean()

    return PLCC, SRCC, KRCC, RMSE, R2, l1


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

def metric_cal_loaded(preds_dict, labels_dict, config):

    preds, labels = [], []

    for key, pred in preds_dict.items():

        # ans, su_flag = pred_trans(pred)
        try:
            gt = labels_dict[key]
            # if gt > 66.9:
                #66.9
            labels.append(gt)
            preds.append(float(pred))  
        except:
            print('video: '+ key + ' is not used')
            continue        
    # print(preds)


    

    if config.no_fit:
        PLCC, SRCC, KRCC, RMSE, R2, l1 = performance_no_fit(np.array(labels), np.array(preds))
    else:
        PLCC, SRCC, KRCC, RMSE, R2, l1 = performance_fit(np.array(labels), np.array(preds))


    print(f"PLCC: {PLCC}, SRCC: {SRCC}, KRCC: {KRCC}, RMSE: {RMSE}, R2: {R2}, L1: {l1}")
    return PLCC, SRCC, KRCC, RMSE

def metric_cal(pred_file, label_file, config):
    preds_dict = process(pred_file)

    labels_pd = pd.read_csv(label_file)

    if config.dataset == "Konvid-1k" or config.dataset == "YT-ugc":
        labels_pd.columns = ["name", "mos"]
        labels_dict = {str(key): value * 20 for key , value in zip(list(labels_pd['name'].values), list(labels_pd['mos'].values))}
    else:
        labels_dict = {key: value for key , value in zip(list(labels_pd['name'].values), list(labels_pd['mos'].values))}

    PLCC, SRCC, KRCC, RMSE = metric_cal_loaded(preds_dict, labels_dict, config)










if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset', type=str, help='Konvid-1k or LSVQ or LBVD or LIVE-VQC or LIVE-Gaming or YT-ugc')
    parser.add_argument('--in_file', type=str, help='The score result')
    parser.add_argument('--no_fit', action='store_true')
    config = parser.parse_args()

    meta_data_path = "data/origin_data/"
    label_file_dict = {"Konvid-1k": "KoNViD_1k_mos.csv", "LSVQ_1080p": "LSVQ_whole_test_1080p.csv", "LSVQ": "LSVQ_whole_test.csv", "LBVD": "LBVD_whole.csv", "LIVE-VQC": "LIVE-VQC_.csv", "LIVE-Gaming": "LIVE_Gaming.csv", "YT-ugc": "youtube_ugc_whole.csv"}
    label_file = meta_data_path + label_file_dict[config.dataset]
    metric_cal(config.in_file, label_file, config)

