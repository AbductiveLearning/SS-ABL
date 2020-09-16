# Semi-Supervised Abductive Learning for Theft Judicial Sentencing

This is the repository for holding the sample code of the Semi-Supervised Abductive Learning framework for Theft Judicial Sentencing experiments in _Semi-Supervised Abductive Learning and Its Application to Theft Judicial Sentencing_ in ICDM 2020.

**This code is only tested in Linux environment.**

## Dependency

- Python 3.6

- tensorflow 1.12.0


## Running Code

### Set Data File Path

```python3
# Before running code, we should set data file path first.
# By default, all data files are in "./data/"

# Data file path's setting codes are lies on line 207 - 212 in "ss_abl_model.py".
# Bert Pretrain json file name
pretrain_filename = "0_0.10.json"

# Sentence model supervised traning data file path
pretrain_money_filename = "./data/0_0.10.csv" 

# Unlabeled data file name
abl_train_filename = "1_0.90.json"
abl_train_money_filename = "1_0.90.csv"

# Bert test data file name
test_filename = "10.json"
# Sentence model test data file name
test_money_filename = "10.csv"
```

```bash
unzip data/dataset.zip
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
python ss_abl_model.py
```

### Parameters

Our model's parameters are listed below.

```python3
# Model's Parameter and default value
abl_max_change_num = 2 # The max number of label could be changed on abduction

abl_times = 1 # The model traning iteration number
rule_file_path = "rule_test_file.txt" # abduction rule file path

pretrain_bert_train_epochs = 16 # The epoch number of BERT training on Supervised data
pretrain_sentence_model_times = 3 # The epoch number of sentence model traning on Supervised data
abl_bert_train_epochs = 1 # The epoch number per iteration of BERT traning on abduction process
abl_sentence_model_times = 3 # The epoch number per iteration of sentence model traning on abduction process

```
