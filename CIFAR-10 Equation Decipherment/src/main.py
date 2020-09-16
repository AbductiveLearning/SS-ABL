import os
import numpy as np
import random
import keras
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from functools import partial
from keras import optimizers
from models import NN_model
from itertools import combinations

import argparse

DEBUG = True


def get_img_data(src_path, labels, shape=(28, 28, 1)):
    print("\n** Now getting all images **")
    #image
    X = []
    #label
    Y = []
    h = shape[0]
    w = shape[1]
    d = shape[2]
    #index = [0,1,2,3]
    for (index, label) in enumerate(labels):
        label_folder_path = os.path.join(src_path, label)
        for p in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, p)
            if d == 1:
                mode = 'I'
            else:
                mode = 'RGB'
            image = Image.open(image_path).convert(mode).resize((h, w))
            X.append((np.array(image) - 127.5) / 255.)
            Y.append(index)

    X = np.array(X)
    Y = np.array(Y)

    index = np.array(list(range(len(X))))
    np.random.seed(random_seed)
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]

    assert (len(X) == len(Y))
    print("Total data size is :", len(X))
    # normalize
    X = X.reshape(-1, h, w, d)
    Y = np_utils.to_categorical(Y, num_classes=len(labels))
    return X, Y


def net_model_pretrain(train_src_path, test_src_path, labels, src_data_name, src_data_file, shape=(28, 28, 1), label_rate=0.2, pretrain_epochs=10):
    print("\n** Now use label data to pretrain the model **")

    X, Y = get_img_data(train_src_path, labels, shape)
    num = int(len(X) * label_rate)
    labeled_X = X[:num]
    labeled_Y = Y[:num]
    unlabeled_X = X[num:]
    print("There are %d pretrain images" % len(labeled_X))
    file_name = '%s_pretrain_weights.hdf5' % src_data_name
    if os.path.exists(file_name):
        print("Pretrain file exists, skip pretrain step.")
        return labeled_X,labeled_Y,unlabeled_X

    # Load CNN
    model = NN_model.get_cifar10_net(len(labels), input_shape=shape)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',#mean_squared_error
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(
        width_shift_range=2,
        height_shift_range=2,
        horizontal_flip=True)

    print('Pretraining...')
    for i in range(pretrain_epochs+1):
        print(i,"epoches")
        datagen.fit(labeled_X)
        model.fit_generator(datagen.flow(labeled_X, labeled_Y, batch_size=64),
                                steps_per_epoch=labeled_X.shape[0]/64,
                                epochs=1)
        #model.fit(labeled_X, labeled_Y, epochs=1, batch_size=64)
        if i%10==0:
            test_nn_model(model, test_src_path, labels, shape)
    model.save_weights(file_name)
    print("Model saved to ", file_name)
    # Get file test data if need testing
    abduced_map = {0:0, 1:1, 2:'+', 3:'='}
    if src_data_file is not None:
        _,_,_,_,equations_true_by_len_test,equations_false_by_len_test = get_file_data(src_data_file, 1, 0)
        for equations_type, (equations_true, equations_false) in enumerate(zip(equations_true_by_len_test, equations_false_by_len_test)):
            #for each length of test equations
            accuracy = validation(model, equations_true, equations_false, labels, abduced_map, shape)
            print("The result of testing length %d equations is:%f\n" %
                    (equations_type + 5, accuracy))
    return labeled_X,labeled_Y,unlabeled_X

def divide_equation_by_len(equations):
    '''
    Divide equations by length
    equations has alreadly been sorted, so just divide it by equation's length
    '''
    equations_by_len = list()
    start = 0
    for i in range(1, len(equations) + 1):
        #print(len(equations[i]))
        if i == len(equations) or len(equations[i]) != len(equations[start]):
            equations_by_len.append(equations[start:i])
            start = i
    return equations_by_len


def split_equation(equations_by_len, prop_train, prop_val):
    '''
    Split the equations in each length to training and validation data according to the proportion
    '''
    train = []
    val = []
    all_prop = prop_train + prop_val
    for equations in equations_by_len:
        random.shuffle(equations)
        train.append(equations[:len(equations) // all_prop * prop_train])
        val.append(equations[len(equations) // all_prop * prop_train:])
        #print(len(equations[:len(equations)//all_prop*prop_train]))
        #print(len(equations[len(equations)//all_prop*prop_train:]))
    return train, val

def abduce_from_list(eq, pos):
    if len(pos)==0:
        if check_eq_consistent(eq):
            return eq
        return None
    eq_copy = eq.copy()
    for c in [0,1,'+','=']:
        eq_copy[pos[0]] = c
        abduced_eq = abduce_from_list(eq_copy, pos[1:])
        if abduced_eq is not None:
            return abduced_eq
    return None

def abduce_eq_npos(eq, change_num):
    if change_num == 0:
        if check_eq_consistent(eq):
            return eq
        else:
            return None
    pos_list = list(combinations(range(len(eq)), change_num))
    for pos in pos_list:
        abduced_eq = abduce_from_list(eq, pos)
        if abduced_eq is not None:
            return abduced_eq
    return None

def get_abduced_eq(eq, max_change_num):
    for change_num in range(max_change_num+1):
        abduced_eq = abduce_eq_npos(eq, change_num)
        if abduced_eq is not None:
            return abduced_eq
    return None #failed

def get_abduced_eqs(exs, mapping):
    abduced_equations_labels = []
    abduced_ids = []
    for i, ex in enumerate(exs):
        # Conver according to mapping
        equation = [mapping[c] for c in ex]
        # Abduce a single equation
        abduced_eq = get_abduced_eq(equation, 2)
        if abduced_eq is None:
            #print("Cannot abduce:", equation)
            continue
        #print("Abduce success:", equation, abduced_eq)
        # Convert back
        mapping_r = {k:v for v,k in mapping.items()}
        abduced_eq = [mapping_r[c] for c in abduced_eq]
        abduced_equations_labels.append(abduced_eq)
        abduced_ids.append(i)
    print("There are %d equations, and we abduce %d equations"%(len(exs),len(abduced_ids)))
    print(abduced_equations_labels)
    return abduced_equations_labels, np.array(abduced_ids)

def check_eq_consistent(sign_list, sys = 2):
    # Valid equation's length is bigger or equal to 5
    if (len(sign_list) < 5):
        return False

    # Equation has only one '+' and '='
    if (sign_list.count('+') != 1):
        return False
    ad_p = sign_list.index('+')

    if (sign_list.count('=') != 1):
        return False
    eq_p = sign_list.index('=')

    # The place_index of '=' is bigger than the place_index of '+'
    if (ad_p > eq_p):
        return False
    
    a = sign_list[:ad_p]
    b = sign_list[ad_p + 1: eq_p]
    c = sign_list[eq_p + 1:]
    len_a = len(a)
    len_b = len(b)
    len_c = len(c)

    # Numbers have at least one digit
    if (len_a == 0):
        return False
    if (len_b == 0):
        return False
    if (len_c == 0):
        return False
    
    # Numbers have not leading zeros
    if (len_a > 1 and a[0] == 0):
        return False
    if (len_b > 1 and b[0] == 0):
        return False
    if (len_c > 1 and c[0] == 0):
        return False

    # Addition simulation
    a = a[::-1]
    b = b[::-1]
    c = c[::-1]
    result_c = []
    max_len = max(len(a), len(b))
    auxiliary_v = 0
    for i in range(max_len):
        t_a = 0
        t_b = 0
        if (len_a > i):
            t_a = a[i]
        if (len_b > i):
            t_b = b[i]
        t_c = auxiliary_v + t_a + t_b
        result_c.append(t_c % sys)
        t_c //= sys

    # Check auxiliary value
    if (t_c > 0):
        result_c.append(t_c)

    # Check length
    if (len(c) != len(result_c)):
        return False

    # Check value
    for t_c, r_c in zip(c, result_c):
        if (t_c != r_c):
            return False
    return True
    
def check_eqs_consistent(exs, mapping):
    for ex in exs:
        eq = [mapping[c] for c in ex]
        if check_eq_consistent(eq)==False:
            return False
    return True


def get_equations_labels(model,
                         equations,
                         labels,
                         abduced_map=None,
                         shape=(28, 28, 1)):
    '''
    Get the model's abduced output through abduction
    model: NN model
    equations: equation images
    labels: [0,1,10,11] now  only use len(labels)
    maps: abduced map like [0:'+',1:'=',2:0,3:1] if None, then try all possible mappings
    no_change: if True, it indicates that do not abduce, only get rules from equations
    shape: shape of image
    '''
    h = shape[0]
    w = shape[1]
    d = shape[2]
    
    exs = []
    for e in equations:
        exs.append(np.argmax(model.predict(e.reshape(-1, h, w, d)), axis=1).tolist())
    print("\n\nThis is the model's current label:")
    print(exs)

    # Check consistency
    is_consistent = check_eqs_consistent(exs, abduced_map)
    if is_consistent == True:
        return exs, np.array(range(1,len(equations)))

    # Find the possible wrong position in symbols and Abduce the right symbol through logic module
    abduced_equations_labels, abduced_ids = get_abduced_eqs(exs, abduced_map)

    return abduced_equations_labels, abduced_ids

def get_file_data(src_data_file, prop_train, prop_val):
    print("Loading equations images...")
    with open(src_data_file, 'rb') as f:
        equations = pickle.load(f)
    input_file_true = equations['train:positive']
    input_file_false = equations['train:negative']
    input_file_true_test = equations['test:positive']
    input_file_false_test = equations['test:negative']

    equations_true_by_len = divide_equation_by_len(input_file_true)
    equations_false_by_len = divide_equation_by_len(input_file_false)
    equations_true_by_len_test = divide_equation_by_len(input_file_true_test)
    equations_false_by_len_test = divide_equation_by_len(input_file_false_test)
    #train:validation:test = prop_train:prop_val
    equations_true_by_len_train, equations_true_by_len_validation = split_equation(
        equations_true_by_len, prop_train, prop_val)
    equations_false_by_len_train, equations_false_by_len_validation = split_equation(
        equations_false_by_len, prop_train, prop_val)

    for equations_true in equations_true_by_len:
        print("There are %d true training and validation equations of length %d"
              % (len(equations_true), len(equations_true[0])))
    for equations_false in equations_false_by_len:
        print("There are %d false training and validation equations of length %d"
              % (len(equations_false), len(equations_false[0])))
    for equations_true in equations_true_by_len_test:
        print("There are %d true testing equations of length %d" %
              (len(equations_true), len(equations_true[0])))
    for equations_false in equations_false_by_len_test:
        print("There are %d false testing equations of length %d" %
              (len(equations_false), len(equations_false[0])))

    return (equations_true_by_len_train, equations_true_by_len_validation,
            equations_false_by_len_train, equations_false_by_len_validation,
            equations_true_by_len_test, equations_false_by_len_test)

def flatten(l):
	return [item for sublist in l for item in sublist]

def abduce_and_train(base_model, equations_true, labeled_X, labeled_Y, labels, abduced_map, shape, SELECT_NUM, BATCHSIZE, NN_EPOCHS):
    h = shape[0]
    w = shape[1]
    d = shape[2]
    #Randomly select several equations
    select_index = np.random.randint(len(equations_true), size=SELECT_NUM)
    select_equations = np.array(equations_true)[select_index]

    #Abduce
    abduced_equations_labels, abduced_ids = get_equations_labels(base_model, select_equations, labels, abduced_map, shape)
    # Can not abduce all equations
    if len(abduced_equations_labels)==0:
        return False

    train_pool_X = np.concatenate(select_equations[abduced_ids]).reshape(-1, h, w, d)
    train_pool_Y = np_utils.to_categorical(flatten(abduced_equations_labels), num_classes=len(labels))  # Convert the symbol to network output
    train_pool_X = np.append(train_pool_X, labeled_X, axis=0)
    train_pool_Y = np.append(train_pool_Y, labeled_Y, axis=0)
    assert (len(train_pool_X) == len(train_pool_Y))
    print("\nTrain pool size is :", len(train_pool_X))
    print("Training...")
    #cifar10  data augmentation
    
    datagen = ImageDataGenerator(
        width_shift_range=2,
        height_shift_range=2,
        horizontal_flip=True)
    datagen.fit(train_pool_X)
    base_model.fit_generator(datagen.flow(train_pool_X, train_pool_Y, batch_size=64),
                            steps_per_epoch=train_pool_X.shape[0]/64,
                            epochs=NN_EPOCHS)
    '''
    base_model.fit(train_pool_X,
                    train_pool_Y,
                    batch_size=BATCHSIZE,
                    epochs=NN_EPOCHS)
                    #,verbose=0)
    '''
    return True

def validation(base_model, equations_true_val, equations_false_val, labels, abduced_map, shape):
    print("Validation: ")

    h = shape[0]
    w = shape[1]
    d = shape[2]
    
    correct_cnt = 0
    total_cnt = 0
    acc = 0
    for e in equations_true_val:
        total_cnt += 1
        eq = np.argmax(base_model.predict(e.reshape(-1, h, w, d)), axis=1).tolist()
        #print("\n\nThis is the model's current label:")
        #print(eq)
        if check_eqs_consistent([eq], abduced_map):
            correct_cnt += 1
    acc = correct_cnt/total_cnt
    print("True equations validation accuracy:", acc)
    for e in equations_false_val:
        total_cnt += 1
        eq = np.argmax(base_model.predict(e.reshape(-1, h, w, d)), axis=1).tolist()
        #print("\n\nThis is the model's current label:")
        #print(eq)
        if check_eqs_consistent([eq], abduced_map)==False:
            correct_cnt += 1
    acc = correct_cnt/total_cnt
    print("Total equations validation accuracy:", acc)
    return acc


def test_nn_model(model, src_path, labels, input_shape):
    print("\nNow test the NN model")
    X, Y = get_img_data(src_path, labels, input_shape)
    print('\nTesting...')
    loss, accuracy = model.evaluate(X, Y, verbose=0)
    print('Neural network perception accuracy: ', accuracy)


def main_func(labels, src_data_name, src_data_file, test_src_path, labeled_X, labeled_Y, shape, args):
    #EQUATION_MAX_LEN = 7 #Only learn the equations of length 5-7
    EQUATION_LEAST_LEN = args.EQUATION_LEAST_LEN
    EQUATION_MAX_LEN = args.EQUATION_MAX_LEN
    #SELECT_NUM = 10 #Select 10 equations to abduce rules
    SELECT_NUM = args.SELECT_NUM
    #
    ## Proportion of train and validation = 9:1
    #PROP_TRAIN = 9
    PROP_TRAIN = args.PROP_TRAIN
    #PROP_VALIDATION = 1
    PROP_VALIDATION = args.PROP_VALIDATION

    #CONDITION_CNT_THRESHOLD = 10       #If the condition has been satisfied 10 times, the start validation
    CONDITION_CNT_THRESHOLD = args.CONDITION_CNT_THRESHOLD
    #NEXT_COURSE_ACC_THRESHOLD = 0.8  #If the validation accuracy of a course higher than the threshold, then go to next course
    NEXT_COURSE_ACC_THRESHOLD = args.NEXT_COURSE_ACC_THRESHOLD

    #NN_BATCHSIZE = 64    # Batch size of neural network
    NN_BATCHSIZE = args.NN_BATCHSIZE
    #NN_EPOCHS = 10       # Epochs of neural network
    NN_EPOCHS = args.NN_EPOCHS

    # Get NN model and compile
    base_model = NN_model.get_cifar10_net(len(labels), shape)
    base_model.load_weights('%s_pretrain_weights.hdf5' % src_data_name)
    base_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Get file data
    equations_true_by_len_train,equations_true_by_len_validation,equations_false_by_len_train,\
    equations_false_by_len_validation,equations_true_by_len_test,equations_false_by_len_test = get_file_data(src_data_file, PROP_TRAIN, PROP_VALIDATION)

    abduced_map = {0:0, 1:1, 2:'+', 3:'='}
    # Start training / for each length of equations
    condition_cnt = 0
    for equations_type in range(EQUATION_LEAST_LEN - 5, EQUATION_MAX_LEN - 4):
        if equations_type==1: # Skip lenggth 6 equations
            continue
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("LENGTH: ", 5 + equations_type)#, " to ", 5 + equations_type + 1)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        equations_true = equations_true_by_len_train[equations_type]
        equations_false = equations_false_by_len_train[equations_type]
        equations_true_val = equations_true_by_len_validation[equations_type]
        equations_false_val = equations_false_by_len_validation[equations_type]
        #equations_true.extend(equations_true_by_len_train[equations_type + 1])
        #equations_false.extend(equations_false_by_len_train[equations_type + 1])
        #equations_true_val.extend(equations_true_by_len_validation[equations_type + 1])
        #equations_false_val.extend(equations_false_by_len_validation[equations_type + 1])
        # Initial test before training
        init_acc = validation(base_model, equations_true_val, equations_false_val, labels, abduced_map, shape)
        best_acc = init_acc
        condition_cnt = 0 #the times of loop 
        while True:
            condition_cnt += 1
            # Abduce and train NN
            select_num = condition_cnt*20 #SELECT_NUM 
            if select_num > 300:
                select_num = 300
            abduce_result = abduce_and_train(base_model, equations_true, labeled_X, labeled_Y, labels, abduced_map, shape, select_num, NN_BATCHSIZE, NN_EPOCHS)
            if abduce_result == False:
                continue

            # Test every CONDITION_CNT_THRESHOLD times
            if condition_cnt % CONDITION_CNT_THRESHOLD==0:
                test_nn_model(base_model, test_src_path, labels, shape)
                accuracy = validation(base_model, equations_true_val, equations_false_val, labels, abduced_map, shape)
                # Next course(larger than init accuracy and lower than the best accuracy) or continue learning
                if accuracy <= best_acc and accuracy > init_acc: #NEXT_COURSE_ACC_THRESHOLD:
                    print("Go to next course")
                    #base_model.save_weights('%s_nlm_weights_%d.hdf5' % (src_data_name, equations_type))
                    break
                else:
                    best_acc = accuracy
                    print("Current accuracy %f. Not reach the best. Continue training."%(accuracy))

    # Final test 
    test_nn_model(base_model, test_src_path, labels, shape)
    for equations_type, (equations_true, equations_false) in enumerate(
            zip(equations_true_by_len_test, equations_false_by_len_test)):
        #for each length of test equations
        accuracy = validation(base_model, equations_true, equations_false, labels, abduced_map, shape)
        print("The result of testing length %d equations is:%f\n" %
                (equations_type + 5, accuracy))

    return base_model


def arg_init():
    parser = argparse.ArgumentParser()

    #EQUATION_LEAST_LEN = 5 #Only learn the equations of length 5-7
    parser.add_argument(
        '--ELL',
        dest="EQUATION_LEAST_LEN",
        metavar='EQUATION_LEAST_LEN',
        type=int,
        default=5,
        help='Equation least (minimum) length for training, default is 5')

    #EQUATION_MAX_LEN = 7 #Only learn the equations of length 5-7
    parser.add_argument(
        '--EML',
        dest="EQUATION_MAX_LEN",
        metavar='EQUATION_MAX_LEN',
        type=int,
        default=7,
        help='Equation max length for training, default is 7')

    #SELECT_NUM = 20 #Select 20 equations to abduce rules
    parser.add_argument(
        '--SN',
        dest="SELECT_NUM",
        metavar='SELECT_NUM',
        type=int,
        default=50,
        help=
        'Every time pick SELECT_NUM equations to abduce rules, default is 50')

    # Proportion of train and validation = 3:1
    #PROP_TRAIN = 9
    parser.add_argument(
        '--PT',
        dest="PROP_TRAIN",
        metavar='PROP_TRAIN',
        type=int,
        default=9,
        help='Proportion of train and validation rate, default PROP_TRAIN is 9'
    )
    #PROP_VALIDATION = 1
    parser.add_argument(
        '--PV',
        dest="PROP_VALIDATION",
        metavar='PROP_VALIDATION',
        type=int,
        default=1,
        help=
        'Proportion of train and validation rate, default PROP_VALIDATION is 1'
    )

    #CONDITION_CNT_THRESHOLD = 10       #If the condition has been satisfied 5 times, the start validation
    parser.add_argument('--CPT', dest="CONDITION_CNT_THRESHOLD", metavar='CONDITION_CNT_THRESHOLD', type=int, default=10, \
      help='If the condition has been satisfied CONSISTENT_PERCENTAGE_THRESHOLD times, the start validation, default is 10')
    #NEXT_COURSE_ACC_THRESHOLD = 0.82  #If the validation accuracy of a course higher than the threshold, then go to next course
    parser.add_argument('--NCAT', dest="NEXT_COURSE_ACC_THRESHOLD", metavar='NEXT_COURSE_ACC_THRESHOLD', type=float, default=0.86, \
      help='If the validation accuracy of a course higher than the threshold, then go to next course, default is 0.82')

    #NN_BATCHSIZE = 64    # Batch size of neural network
    parser.add_argument('--NB',
                        dest="NN_BATCHSIZE",
                        metavar='NN_BATCHSIZE',
                        type=int,
                        default=64,
                        help='Batch size of neural network, default is 64')
    #NN_EPOCHS = 1       # Epochs of neural network
    parser.add_argument('--NE',
                        dest="NN_EPOCHS",
                        metavar='NN_EPOCHS',
                        type=int,
                        default=1,
                        help='Epochs of neural network, default is 1')

    parser.add_argument('--src_dir',
                        metavar='dataset dir',
                        type=str,
                        default="../dataset",
                        help="Where store the dataset")
    parser.add_argument('--train_src_data_name',
                        type=str,
                        default="cifar10_images_train",
                        help="Dataset name(train)")
    parser.add_argument('--test_src_data_name',
                        type=str,
                        default="cifar10_images_test",
                        help="Dataset name(test")
    parser.add_argument('--height', type=int, default=32, help='Img height')
    parser.add_argument('--weight', type=int, default=32, help='Img weight')
    parser.add_argument('--channel',
                        type=int,
                        default=3,
                        help='Img channel num')

    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Pretrain_epochs, default is 100')
    parser.add_argument(
        '--src_data_file',
        type=str,
        default="cifar10_equation_data_train_len_7_test_len_7_sys_2_.pk",
        help="This file is generated by equation_generator.py")
    parser.add_argument('--label_rate', type=float, default=0.2, help="Label rate, default is 0.2")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # here the "labels" are just dir names for storing handwritten images of the 4 symbols
    labels = ['0', '1', '3', '7']
    args = arg_init()
    src_dir = args.src_dir
    train_src_data_name = args.train_src_data_name
    test_src_data_name = args.test_src_data_name
    input_shape = (args.height, args.weight, args.channel)
    src_data_file = args.src_data_file
    label_rate = args.label_rate
    train_src_path = os.path.join(src_dir, train_src_data_name)
    test_src_path = os.path.join(src_dir, test_src_data_name)

    # Pre-train for the CNN
    labeled_X, labeled_Y, unlabeled_X = net_model_pretrain(train_src_path=train_src_path,
                       test_src_path=test_src_path,
                       labels=labels,
                       src_data_name=train_src_data_name,
                       src_data_file=src_data_file,
                       shape=input_shape,
                       label_rate=label_rate,
                       pretrain_epochs=args.pretrain_epochs)

    # Semi-Supervised Abductive Learing main function
    model = main_func(labels=labels,
                          src_data_name=train_src_data_name,
                          src_data_file=src_data_file,
                          test_src_path=test_src_path,
                          labeled_X=labeled_X, 
                          labeled_Y=labeled_Y,
                          shape=input_shape,
                          args=args)
    