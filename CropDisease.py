import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import time

fileRoot = "/home/zg2358/"
zipsize =128

print("Tensorflow version: ", tf.__version__, "\n")
print("cv2 version: ", cv2.__version__, "\n")
time.sleep(1)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), "\n")
time.sleep(1)

print ("path setting")
path_healthy = "train/healthy_wheat/"
path_leaf = "train/leaf_rust/"
path_stem = "train/stem_rust/"
print ("path set\n")
time.sleep(1)

print ("data loading block declaring")
def blur(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel=kernel)
def preprocess(res):
    zipsize = 256
    res = cv2.resize(res, dsize=(zipsize, zipsize))
    res = blur(res)
    res = res [64:192,64:192]
    return res
def loadData():
    data_img = []
    data_label = []
    for file in os.listdir(fileRoot + path_healthy):
        img = cv2.imread(fileRoot + path_healthy + file)
        res = preprocess(img)
        data_img.append(res)
        data_label.append(2)
        data_img.append(cv2.flip(res, 1))
        data_label.append(2)
        data_img.append(cv2.flip(res, 0))
        data_label.append(2)
        data_img.append(cv2.flip(res, -1))
        data_label.append(2)

    for file in os.listdir(fileRoot + path_leaf):
        img = cv2.imread(fileRoot + path_leaf + file)
        if img is None:
            print(file)
            continue
        res = preprocess(img)
        data_label.append(0)
        data_img.append(res)
        data_img.append(cv2.flip(res, 1))
        data_label.append(0)
        data_img.append(cv2.flip(res, 0))
        data_label.append(0)
        data_img.append(cv2.flip(res, -1))
        data_label.append(0)

    for file in os.listdir(fileRoot + "train/stem_rust/"):
        img = cv2.imread(fileRoot + "train/stem_rust/" + file)
        res = preprocess(img)
        data_img.append(res)
        data_label.append(1)
        data_img.append(cv2.flip(res, 1))
        data_label.append(1)
        data_img.append(cv2.flip(res, 0))
        data_label.append(1)
        data_img.append(cv2.flip(res, -1))
        data_label.append(1)
        
    for i in range(len(data_img)):
        data_img[i] = data_img[i] / 255
    data_img = np.array(data_img)
    return data_img, data_label
print ("data loading block declared\n")
time.sleep(1)

print("model block declaring")
def trainTestSplit(data_img,data_label):

    X_train, X_test, y_train, y_test = train_test_split(data_img,data_label,test_size = 0.3)
    y_train =np.array(y_train)
    y_test =np.array(y_test)
    return X_train, X_test, y_train, y_test

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

def trainModel(X_train, X_test, y_train, y_test):
   
    with tf.device('/gpu:0'):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(zipsize, zipsize, 3)))
        tf.keras.layers.Dropout(0.2)
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        tf.keras.layers.Dropout(0.2)
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        tf.keras.layers.Dropout(0.2)
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        tf.keras.layers.Dropout(0.2)
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        tf.keras.layers.Dropout(0.2)
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(3,activation='softmax'))

        model.compile(optimizer='Adamax',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train
        history = model.fit(X_train, y_train,batch_size = 16, epochs=20, use_multiprocessing = True)

        # test
        #test_loss, test_acc = model.evaluate(X_test,y_test, batch_size = 16, verbose=2)
        #print(test_acc)
        return model
print("model block declared\n")
time.sleep(1)

print("Data loading")
data_img,data_lable = loadData()
print("Data Loaded\n")
time.sleep(1)


#X_train, X_test, y_train, y_test = trainTestSplit(data_img,data_lable)

print("model training")
#model = trainModel(X_train, X_test, y_train, y_test)
data_label = np.array(data_lable)
model = trainModel(data_img,[],data_label,[])
print("model trained\n")
time.sleep(1)

print("testing data loading\n")
def loadTest():
    test = []
    name = []
    for file in os.listdir(fileRoot + "test/"):
        img = cv2.imread(fileRoot + "test/" + file)
        res = preprocess(img)
        test.append(res)
        name.append(file)
    for i in range(len(test)):
        test[i] = test[i] / 255
    test = np.array(test)
    result = model.predict_proba(test)
    return name, result
name,result = loadTest()
print("testing data loaded\n")
time.sleep(1)

import pandas as pd

print("submission.csv modifing")
output = []
for i in range(len(result)):
    output.append(np.append(result[i],name[i][0:6]).tolist())
my_df = pd.DataFrame(output)
# change order of columns of csv file so the name is in the first column
#print (my_df)
my_df = my_df[[3, 0, 1, 2]]
my_df.columns = ['ID', 'leaf_rust', 'stem_rust', 'healthy_wheat']
my_df.to_csv('submission.csv', index=False)

print("submission.csv modified\n")
time.sleep(10)
















