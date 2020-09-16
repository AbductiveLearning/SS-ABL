# coding=utf-8

import csv
from sklearn.utils import shuffle
import json

def process_money(money, seg):
    if money >= seg:
        m = (money+seg/2)//seg*seg
    elif money <= 499:
        m = 499
    elif money <= 999:
        m = 999
    else:
        m = money
    return m

def read_csv(csv_file):
    with open(csv_file, 'r', encoding="utf-8") as f:
        data = []
        labels = []
        filenames = []
        ahs = []
        attrs = []
        
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            [filename, ah, money, damage, attitude, surrender, again, young, forgive, tool, room, theft,  year_num, probation, money_num] = row
            if float(money) < 0.1:
                continue
            if int(year_num) == 0:
                continue
            if float(money) >= 30000:
                continue
            money = process_money(float(money), 1000)
            data.append([money])
            attrs.append([int(damage), int(attitude), int(surrender), int(again), int(young), int(forgive), int(tool), int(room), int(theft)]) 
            labels.append(int(year_num))
            filenames.append(filename)
            ahs.append(ah)

    filenames, ahs, data, labels, attrs = shuffle(filenames, ahs, data, labels, attrs, random_state=5)
    return filenames, ahs, data, labels, attrs

def getJson(JsonPath, ah):
    fin = open(JsonPath, 'r', encoding='utf-8')
    for line in fin.readlines():
        judgement = json.loads(line)
        if ah == judgement[0]["ah"]:
            return judgement
    print("Not find judgement in json file according to ah")
    return None

def save_json(JsonPath, data):
    with open(JsonPath, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def load_json(file_path):
    ret = []
    with open(file_path, "r", encoding = "utf-8") as fin:
        for data in fin:
            ret.append(json.loads(data.strip()))
    return ret
