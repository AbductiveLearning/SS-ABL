import csv
from sklearn.utils import shuffle
import json

def read_data_from_csv(csv_file):
    with open(csv_file, 'r', encoding="utf-8") as f:
        data = []
        labels = []
        filenames = []
        ahs = []
        attrs = []
        
        # Read csv and prepare data
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            [filename, ah, money, damage, attitude, surrender, again, young, forgive, tool, room, theft,  year_num, probation, money_num] = row
            #print(row)
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

    # Shuffle
    filenames, ahs, data, labels, attrs = shuffle(filenames, ahs, data, labels, attrs, random_state=5)
    return filenames, ahs, data, labels, attrs

def read_csv_rawdata(csv_file):
    data_list = []
    with open(csv_file, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        next(reader, None)
        for row in reader:
            data_list.append(row)
    return data_list

def read_csv_header(csv_file):
    data_list = []
    with open(csv_file, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = list(reader)[0]
    return header

def write_csv(data_list, header, proportions):
    shuffle(data_list)
    datafile_list = []
    begin = 0
    for idx, probation in enumerate(proportions):
        num = int(probation * len(data_list))
        end = min(begin + num, len(data_list))
        csv_filename = "%d_%.2f.csv" % (idx, probation)
        datafile_list.append(csv_filename)
        with open(csv_filename, "w", encoding="utf-8") as fout:
            writer = csv.writer(fout)  
            writer.writerow(header)
            for row in data_list[begin:end]:
                writer.writerow(row)
    return datafile_list


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

if __name__ == "__main__":
    print("Read 40.csv and 50.csv")
    data_list = read_csv_rawdata("40.csv") + read_csv_rawdata("50.csv")
    header = read_csv_header("40.csv")
    attr_datafile_list = write_csv(data_list, header, [0.1, 0.9])
    print("Generated files : ", attr_datafile_list)

    all_json_data = load_json("label_data_safe.json")
    csv_list = ["10.csv", "40.csv" , "50.csv"] 
    csv_list = attr_datafile_list

    for csv_file in csv_list:
        filenames, ahs, _, _, _ = read_data_from_csv(csv_file)
        json_file = ".".join(csv_file.split(".")[:-1]) + ".json"
        save_data_list = []
        for json_data in all_json_data:
            if json_data[0]["ah"] in ahs:
                filename = filenames[ahs.index(json_data[0]["ah"])]
                json_data[0]["file"] = filename
                save_data_list.append(json_data)
                #print(filename)
        save_json(json_file, save_data_list)
        print("save_json", json_file)

