import json
import random

def splitWithPro(pro = 0.25):
    fin = open(u'./data_4_5/40.json', 'r', encoding='utf-8')
    lines = []
    for line in fin.readlines():
        line = json.loads(line)
        lines.append(line)
    random.shuffle(lines)
    test_set_size = int(len(lines) * pro)
    test1 = lines[:test_set_size]
    test2 = lines[test_set_size:2 * test_set_size]
    test3 = lines[test_set_size*2:test_set_size*3]
    test4 = lines[test_set_size*3:]

    with open('./data/all_data/train_1.json', 'w', encoding='utf-8') as f:
        for line in test1:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

    with open('./data/all_data/train_2.json', 'w', encoding='utf-8') as f:
        for line in test2:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

    with open('./data/all_data/train_3.json', 'w', encoding='utf-8') as f:
        for line in test3:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

    with open('./data/all_data/train_4.json', 'w', encoding='utf-8') as f:
        for line in test4:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

def getDic():
    tags_list = []
    with open('data/all_data/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())

    transToDic = dict()
    for key, value in enumerate(tags_list):
        transToDic[value] = key
    return transToDic,tags_list


def getVecFromJson(fromPath,toPath = u'./data/all_data/pre_vecs.json'):
    tags_list = []
    with open(u'./data/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())

    transToDic = dict()
    for key,value in enumerate(tags_list):
        transToDic[value] = key
    vecs = []
    yaoshu = []
    fin = open(fromPath, 'r', encoding='utf-8')

    for line in fin.readlines():
        file_vec = dict()
        vec = [0] * len(tags_list)
        line = json.loads(line)
        crnt_dic = line[0]
        file_name = str(crnt_dic['file'])+"&"+str(crnt_dic['ah'])
        file_vec[file_name] = []

        for dic in line:
            for lb in dic["label"]:
                idx = transToDic[lb]
                vec[idx] = 1
                yaoshu.append(lb)
        file_vec[file_name] = vec
        vecs.append(file_vec)

    return vecs


def calaAcc(pre_path=u'./data/all_data/vecs.json',true_path=u'./data/all_data/test_true_vecs.json',tags=None):
    preLbs = []
    fin = open(pre_path, 'r', encoding='utf-8')
    for line in fin.readlines():
        line = json.loads(line)
        preLbs.append(line)

        trueLbs = []
    fin = open(true_path, 'r', encoding='utf-8')
    for line in fin.readlines():
        line = json.loads(line)
        trueLbs.append(line)

    import numpy as np
    preLbs = np.array(preLbs)
    trueLbs = np.array(trueLbs)
    for i in range(len(preLbs[0])):
        if tags != None:
            print(tags[i])
        col_pre = preLbs[:,i]
        col_true = trueLbs[:,i]
        print(np.mean(col_pre == col_true))

        print("============================")

def getDifferToFile(truePath = u'./data/all_data/10_new.json',bertPath = u'./tmp/newoutput/output_22453_10_new.json' ):
    vec_bert = getVecFromJson(bertPath)
    vec_true = getVecFromJson(truePath)
    differ_idxs = []
    for idx in range(len(vec_bert)):
        if vec_bert[idx] != vec_true[idx]:
            differ_idxs.append(idx)
    print(len(differ_idxs))
    fin_true = open(truePath, 'r', encoding='utf-8')
    fin_bert = open(bertPath,'r',encoding='utf8')

    diff_arr = []
    for idx,value in enumerate(zip(fin_true.readlines(),fin_bert.readlines())):
        if idx in differ_idxs:
            print("true_attrs")
            print(vec_true[idx])
            print("bert_attrs")
            print(vec_bert[idx])
            temp_arr = []
            value_true = json.loads(value[0])
            value_bert = json.loads(value[-1])
            for dic_true,dic_bert in zip(value_true,value_bert):
                set_bert = set(dic_bert['label'])
                set_true = set(dic_true['label'])
                if set_bert != set_true:
                    output_diff = dict()
                    output_diff['file'] = dic_true['file']
                    output_diff['ah'] = dic_true['ah']
                    output_diff['sentence'] = dic_true['sentence']
                    output_diff['bert_label'] = dic_bert['label']
                    output_diff['true_label'] = dic_true['label']
                    output_diff['bert_pro'] = dic_bert['pro']
                    temp_arr.append(output_diff)
                    print('sentence:'+output_diff['sentence'])
                    print('true_label')
                    print(output_diff['true_label'])
                    pri_vec_1 = []
                    for label in output_diff['true_label']:
                        a, _ = getDic()
                        i = a[label]
                        pri_vec_1.append(output_diff['bert_pro'][i])
                    print('true_label_bert_pro')
                    print(pri_vec_1)
                    print('bert_label')
                    print(output_diff['bert_label'])
                    pri_vec = []
                    for label in output_diff['bert_label']:
                        a,_ = getDic()
                        i = a[label]
                        pri_vec.append(output_diff['bert_pro'][i])
                    print('bert_pro')
                    print(pri_vec)

            diff_arr.append(temp_arr)
    with open(u'./data/all_data/differ.json', 'w', encoding='utf-8') as f:
        for line in diff_arr:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


def find3PosMax(nums):
    max1, max2, max3 = None, None, None
    for num in nums:
        if num < 0:
            continue
        if max1 is None or max1 < num:
            max1, num = num, max1
        if num is None:
            continue
        if max2 is None or num > max2:
            max2, num = num, max2
        if num is None:
            continue
        if max3 is None or num > max3:
            max3 = num
    return max1, max2, max3


