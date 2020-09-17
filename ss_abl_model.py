#coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle as pk

import sys
import csv
from sklearn import tree
from sklearn.utils import shuffle
import pandas as pd
from sklearn import linear_model
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
import json

from data_utils import read_csv, process_money, save_json, load_json, getJson
from tools import getVecFromJson

from bert_class import BERT, get_lastest_ckpt
from sentence_model import SentenceModel

import shutil
import argparse

from abduction_model import SentenceAbduction
from result_utils import ResultRecorder
from judger import get_score

import time
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

is_debug = False

def debug_print(*args):
    if is_debug:
        print(*args)

def my_parse():
    parser = argparse.ArgumentParser(description = 'ABL Tuned Parameters!')
    parser.add_argument('-pretrain_bert_train_epochs', dest='pretrain_bert_train_epochs', \
                        type=int, default=4, help='-pretrain_bert_train_epochs: an integer')
    parser.add_argument('-pretrain_sentence_model_times', dest='pretrain_sentence_model_times', \
                        type=int, default=3, help='-pretrain_sentence_model_times: an integer')
    parser.add_argument('-abl_bert_train_epochs', dest='abl_bert_train_epochs', \
                        type=int, default=1, help='-abl_bert_train_epochs: an integer')
    parser.add_argument('-abl_sentence_model_times', dest='abl_sentence_model_times', \
                        type=int, default=3, help='abl_sentence_model_times: an integer')
    parser.add_argument('-abl_max_change_num', dest='abl_max_change_num', \
                        type=int, default=2, help='abl_max_change_num: an integer')
    parser.add_argument('-rule_file_path', dest='rule_file_path', \
                        type=str, default="rule_file.txt")
    parser.add_argument('-log_dump_file', dest='log_dump_file', \
                        type=str, default="default_log.pk")
    parser.add_argument('-abl_times', dest='abl_times', \
                        type=int, default=1)
    return parser.parse_args()

def selectJson(JsonPath, ahs, JsonOutputPath):
    fout = open(JsonOutputPath, 'w', encoding='utf-8')
    for ah in ahs:
        judgement = getJson(JsonPath, ah)
        json_dicts = json.dumps(judgement, ensure_ascii=False)
        if judgement != None:
            fout.writelines(json_dicts)
            fout.writelines("\n")
    fout.close()

def splitJson(JsonPath, ahs, infile, not_infile):
    fout1 = open(infile, 'w', encoding='utf-8')
    fout2 = open(not_infile, 'w', encoding='utf-8')
    
    fin = open(JsonPath, 'r', encoding='utf-8')
    for line in fin.readlines():
        judgement = json.loads(line)
        json_dicts = json.dumps(judgement, ensure_ascii=False)
        if judgement[0]["ah"] in ahs:
            fout1.writelines(json_dicts)
            fout1.writelines("\n")
        else:
            fout2.writelines(json_dicts)
            fout2.writelines("\n")

    fout1.close()
    fout2.close()

def split_csv(csv_path, ah_list, in_file, not_in_file):
    df = pd.read_csv(csv_path, encoding='gbk')
    data = df[["filename", "ah", "sum", "no_damage_bool", "attitude_bool", "surrender_bool", "again_bool", "young_bool", "forgive_bool",
               "tool_bool", "indoor_bool", "theft_bool", "year_num", "probation", "money_num"]]
    data = np.array(data)
    data = data.tolist()

    rowname = ["filename", "ah", "sum", "no_damage_bool", "attitude_bool", "surrender_bool", "again_bool", "young_bool", "forgive_bool",
               "tool_bool", "indoor_bool", "theft_bool", "year_num", "probation", "money_num"]
    great_file = open(in_file, 'w', newline='')
    other = open(not_in_file, 'w', newline='')
    great_file_csv = csv.writer(great_file)
    great_file_csv.writerow(rowname)

    other_csv = csv.writer(other)
    other_csv.writerow(rowname)
    for row in data:
        if row[1] in ah_list:
            great_file_csv.writerow(row)
        else:
            other_csv.writerow(row)
    great_file.close()
    other.close()

def getDic():
    tags_list = []
    with open('./tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())

    transToDic = dict()
    for key, value in enumerate(tags_list):
        transToDic[value] = key
    return transToDic,tags_list

def get_nlp_label(csv_file, json_file):
    attr_vecs = getVecFromJson(json_file,)
    filenames30, ahs30, data30, labels30, attrs30 = read_csv(csv_file)
    count = 0
    filenames_new, ahs_new, data_new, labels_new, attrs_new = [],[],[],[],[]

    for dic in attr_vecs:
        for idx, value in enumerate(zip(filenames30, ahs30)):
            name = value[0] + "&" + value[-1]
            for k,v in dic.items():
                if k == name:
                    orig_vec = attrs30[idx]
                    if v != orig_vec:
                        count += 1
                        attrs30[idx] = v

                    attrs30[idx] = v
                    filenames_new.append(value[0])
                    ahs_new.append(value[-1])
                    data_new.append(data30[idx])
                    labels_new.append(labels30[idx])
                    attrs_new.append(v)

    return ahs_new, data_new, labels_new, attrs_new

def get_nlp_result(json_file, filenames, ahs, money, labels):
    attr_vecs = getVecFromJson(json_file,)
    filenames_new, ahs_new, data_new, labels_new, attrs_new = [],[],[],[],[]
    
    for dic in attr_vecs:
        for idx, value in enumerate(zip(filenames, ahs)):
            name = value[0] + "&" + value[-1]
            for k, v in dic.items():
                if k == name:
                    filenames_new.append(value[0])
                    ahs_new.append(value[-1])
                    data_new.append(money[idx])
                    labels_new.append(labels[idx])
                    attrs_new.append(v)

    return filenames_new, ahs_new, data_new, labels_new, attrs_new

def get_bert_generate_label(model, context_filename, money_filename, tmp_json_path, tags_list):
    lastest_ckpt = get_lastest_ckpt(model.output_dir + "/checkpoint")
    pred_labels, result_prob = model.predict(context_filename, model.output_dir + "/" + lastest_ckpt)
    model.generate_pred_file(tags_list, context_filename, tmp_json_path, pred_labels, result_prob)
    filenames, ahs, money, labels, _ = read_csv(model.data_dir + '/' + money_filename)
    filenames_new, ahs_new, money_new, labels_new, attrs_new = get_nlp_result(tmp_json_path, filenames, ahs, money, labels)
    return filenames_new, ahs_new, money_new, labels_new, attrs_new

def rmdir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

if __name__ == "__main__":
    recorder = ResultRecorder()

    for arg in sys.argv:
        recorder.write_pair("args", arg)
        
    tags_list = []
    with open('data/tags.txt', 'r', encoding='utf-8') as tagf:
      for line in tagf.readlines():
          tags_list.append(line.strip())

    args = my_parse()
    for arg in args.__dict__:
        recorder.write_pair(arg + "@arg", args.__dict__[arg])
    pretrain_bert_train_epochs = args.pretrain_bert_train_epochs
    pretrain_sentence_model_times = args.pretrain_sentence_model_times
    abl_bert_train_epochs = args.abl_bert_train_epochs
    abl_sentence_model_times = args.abl_sentence_model_times
    abl_max_change_num = args.abl_max_change_num
    rule_file_path = args.rule_file_path
    log_dump_file = args.log_dump_file
    abl_times = args.abl_times

    print("Rule file is :", rule_file_path)
    if rule_file_path == "None":
        rule_file_path = None

    pretrain_filename = "0_0.10.json"
    pretrain_money_filename = "./data/0_0.10.csv"
    abl_train_filename = "1_0.90.json"
    abl_train_money_filename = "1_0.90.csv"
    test_filename = "10.json"
    test_money_filename = "10.csv"

    recorder.write_pair("Method", "ABL")

    pretrain_filenames, pretrain_ahs, pretrain_money, pretrain_labels, pretrain_attrs = read_csv(pretrain_money_filename)

    perception = BERT(bert_path = "./chinese_L-12_H-768_A-12", data_dir = "./data", output_dir = "./abl_model_0", num_train_epochs = pretrain_bert_train_epochs)
    sentence = SentenceModel()
    abductor = SentenceAbduction(sentence, rule_file_path, True)

    perception.train(pretrain_filename)
    lastest_ckpt = get_lastest_ckpt(perception.output_dir + "/checkpoint")
    print("Test BERT:")
    bert_eval_info = perception.eval(test_filename, perception.output_dir + "/" + lastest_ckpt)
    for info in bert_eval_info:
        recorder.write_pair("init_" + info[0], info[1])
    
    sentence.fit(pretrain_money, pretrain_attrs, pretrain_labels, pretrain_sentence_model_times)
    sentence.show_param()
    baseline, rate = sentence.get_param()
    recorder.write_pair("init_baseline", baseline)
    recorder.write_pair("init_rate", rate)

    tmp_json_path = "tmp/abl_predict_%d.json" % (0)
    filenames_new, ahs_new, money_new, labels_new, attrs_new = \
                                    get_bert_generate_label(perception, test_filename, test_money_filename, tmp_json_path, tags_list)

    best_mae, best_mse, _, _ = sentence.test(money_new, attrs_new, labels_new, filenames_new, ahs_new)
    ret_raw, ret_str = get_score("data/" + test_filename, tmp_json_path, "data/tags_for_test.txt")
    recorder.write_pair("init_MAE", best_mae)
    recorder.write_pair("init_MSE", best_mse)
    recorder.write_pair("init_f1_score", ret_str)

    pretrain_bert_train_data = load_json(perception.data_dir + "/" + pretrain_filename)
    

    model_idx = 0
    for t in range(abl_times):
        print("abduction times %d" % t, "model idx %d" % (model_idx))

        tmp_json_path = "tmp/abl_train_%d.json" % (t)
        filenames_new, ahs_new, money_new, labels_new, attrs_new = \
                                            get_bert_generate_label(perception, abl_train_filename, abl_train_money_filename, tmp_json_path, tags_list)
        
        t_b = time.time()
        abductor.set_predict_model(sentence)
        attrs_abduced, labels_abduced, judgement_jsons = abductor.abduce_batch(tmp_json_path, ahs_new, money_new, attrs_new, labels_new, abl_max_change_num)
        t_e = time.time()
        print("Abduce all time cost is : ", t_e - t_b)

        json_save_filename = "abl_retrain_%d.json" % (t + 1)
        save_json(perception.data_dir + "/" + json_save_filename, judgement_jsons + pretrain_bert_train_data)

        perception.output_dir = "./abl_model_%d" % ((model_idx % 10) + 1)
        perception.num_train_epochs = abl_bert_train_epochs
        rmdir(perception.output_dir)
        perception.read_config()
        perception.train(json_save_filename)
        lastest_ckpt = get_lastest_ckpt(perception.output_dir + "/checkpoint")

        print("Test BERT:")
        bert_eval_info = perception.eval(test_filename, perception.output_dir + "/" + lastest_ckpt)
        for info in bert_eval_info:
            recorder.write_pair(info[0], info[1])
        
        filenames_new, ahs_new, money_new, labels_new, attrs_new = \
                                            get_bert_generate_label(perception, abl_train_filename, abl_train_money_filename, tmp_json_path, tags_list)
        sentence.fit(money_new + pretrain_money, attrs_new + pretrain_attrs, labels_new + pretrain_labels, abl_sentence_model_times)
        sentence.show_param()
        baseline, rate = sentence.get_param()
        recorder.write_pair("baseline", baseline)
        recorder.write_pair("rate", rate)

        tmp_json_path = "tmp/abl_predict_%d.json" % (t + 1)
        filenames_new, ahs_new, money_new, labels_new, attrs_new = \
                                        get_bert_generate_label(perception, test_filename, test_money_filename, tmp_json_path, tags_list)

        tmp_mae, tmp_mse, _, _ = sentence.test(money_new, attrs_new, labels_new, filenames_new, ahs_new)
        ret_raw, ret_str = get_score("data/" + test_filename, tmp_json_path, "data/tags_for_test.txt")
        recorder.write_pair("tmp_f1_score", ret_str)
        recorder.write_pair("tmp_MAE", tmp_mae)
        recorder.write_pair("tmp_MSE", tmp_mse)
        
        model_idx += 1
        '''
        if (tmp_mae <= best_mae):
            model_idx += 1
            best_mae = tmp_mae
            print("Model be better!")
        else:
            checkpoint_dir = "./abl_model_%d" % (((model_idx - 1) % 5) + 1)
            lastest_ckpt = get_lastest_ckpt(checkpoint_dir + "/checkpoint")
            perception.output_dir = checkpoint_dir
            perception.init_checkpoint = checkpoint_dir + "/" + lastest_ckpt
            perception.read_config()
            print("Model be worse! Need retrain!")
        '''

        recorder.write_pair("best_MAE", best_mae)
        recorder.write_pair("best_MSE", best_mse)

        print("training times %d" % t)

    recorder.dump(open(log_dump_file, "wb"))

