import pickle
import PIL.Image as Image
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":

    for j in range(1, 6):
        dataName = "cifar-10-batches-py/data_batch_" + str(j)  
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")
        print(Xtr.keys())
        for i in range(0, 10000):
            img = np.reshape(Xtr[b'data'][i], (3, 32, 32)) 
            img = img.transpose(1, 2, 0) 
            picName = 'cifar10_images_train/' +  str(Xtr[b'labels'][i]) +'/'+ str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1)*10000) + '.png' 
            Image.fromarray(img).save(picName)
        print(dataName + " loaded.")

    print("test_batch is loading...")

    testXtr = unpickle("cifar-10-batches-py/test_batch")
    for i in range(0, 10000):
        img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = 'cifar10_images_test/' + str(testXtr[b'labels'][i]) +'/'+str(testXtr[b'labels'][i]) + '_' + str(i) + '.png'
        Image.fromarray(img).save(picName)
    print("test_batch loaded.")

