from itertools import combinations
from sklearn import linear_model

import pickle as pk

from checkrules import CheckRules

from data_utils import load_json
from sentence_model import SentenceModel
import copy
import numpy as np

def check_attr(attr, matched):
    if matched[0]==1 and matched[11]==0 and attr[0]==0:
        return False
    if matched[0]==0 and attr[0]==1:
        return False
    if matched[1] != attr[1]:
        return False
    if matched[2]==0 and attr[2]==1:
        return False
    if matched[2]==1 and matched[10]==0 and attr[2]==0:
        return False
    if matched[3]==1 and matched[10]==0 and attr[3]==0:
        return False
    if matched[3]==0 and attr[3]==1:
        return False
    if matched[4]==1 and matched[12]==1 and attr[4]==0:
        return False
    if matched[4]==0 and attr[4]==1:
        return False
    if matched[5] != attr[5]:
        return False
    if matched[7]==1 and attr[7]==0:
        return False
    if matched[8]==1 and attr[8]==0:
        return False
    if matched[9]==1 and attr[0]==0 and attr[1]==0 and attr[2]==0 and attr[4]==0 and attr[5]==0:
        return False
    
    return True



def attr_convert(attr, pos):
    attr_copy = attr.copy()
    for idx in pos:
        attr_copy[idx] = 1 - attr_copy[idx]
    return attr_copy

class SentenceAbduction:
    def set_predict_model(self, model):
        self.model = model

    def get_matching_re(self, context):
        vec = [0] * len(self.facter_strs)
        if context is None:
            print("Context can not be found!")
            return vec

        for i in range(len(self.facter_strs)):
            facter_str = self.facter_strs[i]
            for facter in facter_str:
                loc = context.find(facter)
                if loc != -1:
                    if i == 7 or i == 8:
                        if abs(context.find(self.not_facter_room_theft_str)-loc) <= 20:
                            vec[i] = 0
                        else:
                            vec[i] = 1
                    elif i == 0:
                        if context[loc-1]=='未':
                            vec[i] = 0
                        else:
                            vec[i] = 1
                    else: 
                         vec[i] = 1
        return vec

    def build_dict(self, json_list):
        context_dict = {}
        for data in json_list:
            ah = data[0]['ah']
            context_dict[ah] = data
        return context_dict

    def get_penalty_type(self, money, attrs):
        [no_damage, attitude, surrender, again, young, forgive, tool, room, theft] = attrs
        if money < self.LARGE:
            if room == 1 or theft == 1:
                return 0
            if money >= 500 and again == 1:
                return 0
        elif money < self.HUGE:
            return 0
        elif money < self.EXTRA_HUGE:
            return 1
        else:
            return 2
        return -1

    def validate(self, money, attrs, month):
        [no_damage, attitude, surrender, again, young, forgive, tool, room, theft] = attrs
        prob = self.check.judge(attrs)
        if prob < 1e-6:
            return False

        penalty_type = self.get_penalty_type(money, attrs)
        if penalty_type == -1:
            return False
        if penalty_type == 0:
            if month <= 3 * 12:
                return True
            elif month >= 3 * 12 and month <= 10 * 12 and room == 1 and money >= 15000:
                return True
            else:
                return False
        if penalty_type == 1:
            if month >= 3 * 12 and month <= 10 * 12:
                return True
            if month <= 3 * 12 and surrender == 1:
                return True
            if month >= 10 * 12 and room == 1 and money >= 150000:
                return True
        if penalty_type == 2:
            if month >= 10 * 12:
                return True
            if month <= 10 * 12 and month >= 3 * 12 and (surrender == 1 or young == 1):
                return True
        return False

    def predict_and_validate(self, X, attrs, target_month):
        Y = self.model.predict(X, attrs)
        for i in range(len(X)):
            [money] = X[i]
            if self.validate(money, attrs[i], target_month[i]) == False:
                Y[i] = -1
        return Y

    def select_abduced_result(self, attrs, months, target_month):
        err = 999
        if len(attrs) == 0:
            return None, None
        assert len(attrs) > 0
        if abs(months[0] - target_month) <= 0.1:
            return attrs[0], months[0]
        for attr, month in zip(attrs, months):
            if month < 0:
                continue
            if abs(month - target_month) < err:
                selected_attr = attr
                selected_month = month
                err = abs(selected_month - target_month)
        return selected_attr, selected_month

    def abduce_npos(self, money, attr, target_month, n, match_res):
        if n == 0:
            [predicted_month] = self.predict_and_validate([money], [attr], [target_month])
            return attr, predicted_month
        pos_list = list(combinations(range(len(attr)), n))
        err = 99
        abduced_attr = None
        abduced_month = -1
        for pos in pos_list:
            new_attr = attr_convert(attr, pos)

            if self.word_match and check_attr(new_attr, match_res) == False:
                continue

            [predicted_month] = self.predict_and_validate([money], [new_attr], [target_month])
            if abs(predicted_month - target_month) < err:
                err = abs(predicted_month - target_month)
                abduced_attr = new_attr.copy()
                abduced_month = predicted_month
        return abduced_attr, abduced_month


    def remove_label(self, content, label, k):
        judgement = []
        modified_thres = 0.9
        for con in content:
            pro = con["pro"]
            if label in con["label"]:
                if pro[k] <= modified_thres:
                    con["label"].remove(label)
                    #print("remove label:", label)
                else:
                    pass#print("should remove but not remove:", label)
            judgement.append(con)
        return judgement

    def get_pro_matrix(self, data):
        pro_matrix = []
        for line in data:
            pro_matrix.append(line["pro"])
        return pro_matrix

    def get_pre_max_n(self, li):
        max_value = max(li)
        index = li.index(max(li))
        li[index] = 0
        return index, max_value

    def add_index_label(self, judgement, index, label):
        line = judgement[index]
        if label not in line['label']:
            line["label"].append(label)
        return judgement

    def get_matching_idxs(self, judgement, k):
        indexs = []
        facter_str = self.facter_strs[k]
        for i,line in enumerate(judgement):
            for facter in facter_str:
                if line['sentence'].find(facter) != -1 and not((k==7 or k==8) and line['sentence'].find(self.not_facter_room_theft_str) != -1) and not(k==2 and line['sentence'].find(self.not_facter_surrender_str1) != -1) and not(k==2 and line['sentence'].find(self.not_facter_surrender_str2) != -1) and not(k==0 and line['sentence'].find(self.order_facter_damage_str[0]) != -1) :
                    indexs.append(i)
                    break
        return indexs

    def add_maxpro_label(self, judgement, matrix, k, label):
        res_list = []
        change = True
        modified_thres = 0.1

        for j in range(len(matrix)):
            res_list.append(matrix[j][k])
        
        indexs = self.get_matching_idxs(judgement, k)
        if len(indexs) != 0:
            #print("add label by matching:", label)
            for idx in indexs:
                judgement = self.add_index_label(judgement, idx, label)
            return judgement

        max_index = res_list.index(max(res_list))
        if res_list[max_index] < modified_thres:
            change = False
            #print("Should add but not add:", label)

        if change:
            #print("add label with max prob:", label)
            if k in [0, 1, 3, 5, 6, 7]:
                indexs = []
                for i in range(2):
                    index, max_value = self.get_pre_max_n(res_list)
                    if max_value >= modified_thres:
                        indexs.append(index)
                for iid in indexs:
                    judgement = self.add_index_label(judgement, iid, label)
                return judgement
            else:
                judgement = self.add_index_label(judgement, max_index, label)
                return judgement
        else:
            return judgement

    def __init__(self, model, rule_file_path, word_match = False):
            self.label2id ={"no_damage": 0, "attitude": 1, "surrender": 2, "again": 3, "young": 4,
                        "forgive": 5, "tool": 6, "indoor": 7, "theft": 8}
            self.id2label = {value:key for key, value in self.label2id.items()}
            self.LARGE = 1000
            self.HUGE = 30000
            self.EXTRA_HUGE = 300000

            self.check = CheckRules(rule_file_path)
            self.model = model
            self.word_match = word_match

            facter_damage_str = ["追回", "退还", "没有给被害人造成损失", "赔偿", "发还", "退缴", "退赔", "追还", "退赃", "返还", "未给被害人造成经济损失","归还"]
            facter_attitude_str = ["如实供述", "主动交代", "认罪", "悔罪", "坦白", "如实交待"]
            facter_surrender_str = ["自首"]
            facter_again_str = ["因犯", "曾因犯罪", "累犯", "前科"]
            facter_young_str = ["未成年", "未满十八周岁",'未满18周岁','不满十八周岁','不满18周岁']
            facter_forgive_str = ["谅解","原谅"]
            facter_tool_str = ["作案工具", "盗窃用的工具", "盗窃用工具"]
            facter_room_str = ["入室", "入户","家中","卧室"]
            facter_theft_str = ["扒窃", "扒取","扒走","上衣","口袋", "衣兜"]
            facter_less_str = ["从轻处罚", "减轻处罚"]
            facter_neg_str = ["不予采纳","不具有","不符","不构成","不属","不认定","不予认定"]
            self.order_facter_damage_str = ["责令"]
            facter_closed_str = ["不公开开庭"]
            self.facter_strs = [facter_damage_str, facter_attitude_str, facter_surrender_str, facter_again_str, facter_young_str, facter_forgive_str, facter_tool_str, facter_room_str, facter_theft_str, facter_less_str, facter_neg_str, self.order_facter_damage_str, facter_closed_str]
            
            self.not_facter_room_theft_str = "多次盗窃、入户盗窃、携带凶器盗窃、扒窃的"
            self.not_facter_surrender_str1 = "犯罪以后自动投案，如实供述自己的罪行的"
            self.not_facter_surrender_str2 = "对于自首的犯罪分子，可以从轻或者减轻处罚"
            
    def __abduce(self, money, attr, target_month, context, max_change_num):
        abduced_attrs = []
        abduced_months = []

        match_res = self.get_matching_re(context)
        for change_num in range(max_change_num+1):
            abduced_attr, abduced_month = self.abduce_npos(money, attr, target_month, change_num, match_res)
            if abduced_month == -1:
                continue
            abduced_attrs.append(abduced_attr)
            abduced_months.append(abduced_month)
        abduced_attr, abduced_month = self.select_abduced_result(abduced_attrs, abduced_months, target_month)
        change = not(abduced_attr is None or attr==abduced_attr)
        if abduced_attr is None:
            print("Ad-hoc", context, money, attr, target_month, "specially: ", match_res, attr)
        return abduced_attr, abduced_month, change

    def abduce(self, money, attr, month, data, max_change_num):
        context = None
        if data is not None:
            context = "".join([d["sentence"].replace(" ", "") for d in data])
            judgement_json = copy.deepcopy(data)
        if data is None:
            return None, None, None
        abduced_attr, abduced_month, change \
            = self.__abduce(money, attr, month, context, max_change_num)

        if change == True:
            assert len(abduced_attr) == len(attr)
            #print("no_damage", "attitude", "surrender", "again", "young", "forgive", "tool", "indoor", "theft")
            #print(attr, month)
            #print(abduced_attr, abduced_month)
            #print(context)
            #print(judgement_json)
            
            pro_matrix = self.get_pro_matrix(judgement_json)
            for k, v in enumerate(abduced_attr):
                if v != attr[k] or (v == attr[k] and v==1):
                    if v == 0:
                        judgement_json = self.remove_label(judgement_json, self.id2label[k], k)
                    elif v == 1: 
                        judgement_json = self.add_maxpro_label(judgement_json, pro_matrix, k, self.id2label[k])
                    else:
                        print("error with label is not 0/1, detail is:", data[0]["ah"], attr, k, v)
        return abduced_attr, abduced_month, judgement_json

    def abduce_batch(self, json_file_path, ahs, moneys, attrs, months, max_change_num):
        abduced_attrs = []
        abduced_months = []
        judgement_jsons = []
        
        context_dict = self.build_dict(load_json(json_file_path))
        for ah, money, attr, month in zip(ahs, moneys, attrs, months):
            data = context_dict.get(ah, None)
            abduced_attr, abduced_month, judgement_json = self.abduce(money, attr, month, data, max_change_num)
            if abduced_attr is None:
                continue

            abduced_attrs.append(abduced_attr)
            abduced_months.append(abduced_month)
            judgement_jsons.append(judgement_json)

        return abduced_attrs, abduced_months, judgement_jsons

    def ad_hoc_test(self, money, attr, month, context, max_change_num):
        return self.__abduce(money, attr, month, context, max_change_num)

