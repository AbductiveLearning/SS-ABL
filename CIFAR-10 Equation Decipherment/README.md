# Abductive Learning for CIFAR-10 Equation Decipherment

This is the sample code of the Semi-Supervised Abductive Learning framework for CIFAR-10 Equation Decipherment experiments in _Semi-Supervised Abductive Learning and Its Application to Theft Judicial Sentencing_ in ICDM 2020.


## Environment dependency

- Python 3.6

- Tensorflow 1.12.0

- Keras 2.2.4

## Running Code

### Prepare Data

```bash
cd dataset 
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz
python process.py
cd ../src
python equation_generator.py
```

### Run Experiment
```
python main.py
```


### Parameters

Our model's parameters are listed below.

```python3
# Model's Parameter and default value

pretrain_epochs = 100 # The epoch number of pre-training on supervised data
NN_BATCHSIZE = 64 # The batch size of neural network while training
abl_max_change_num = 2 # The max number of label could be changed on abduction
label_rate = 0.2 # The label rate of the experiment


```