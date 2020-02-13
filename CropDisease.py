import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
fileRoot = "D:\\ICLR-CropDisease\\dataset\\"


# Load data and preprocess
def loadData():
    data_img = []
    data_label = []

    for file in os.listdir(fileRoot + "train\\healthy_wheat\\"):
        img = cv2.imread(fileRoot + "train\\healthy_wheat\\" + file)
        res = cv2.resize(img, dsize=(128, 128))
        data_img.append(res)
        data_label.append(0)

    for file in os.listdir(fileRoot + "train\\leaf_rust\\"):
        img = cv2.imread(fileRoot + "train\\leaf_rust\\" + file)
        if img is None:
            print(file)
            continue
        res = cv2.resize(img, dsize=(128, 128))
        data_img.append(res)
        data_label.append(1)

    for file in os.listdir(fileRoot + "train\\stem_rust\\"):
        img = cv2.imread(fileRoot + "train\\stem_rust\\" + file)
        res = cv2.resize(img, dsize=(128, 128))
        data_img.append(res)
        data_label.append(2)

    for i in range(len(data_img)):
        data_img[i] = data_img[i] / 255
    data_img = np.array(data_img)
    return data_img, data_label


def trainTestSplit(data_img,data_label):

    X_train, X_test, y_train, y_test = train_test_split(data_img,data_label,test_size = 0.3)
    return X_train, X_test, y_train, y_test

#Create CNN model
# Current : 3 conv layers, 2 pooling, 1 flatten, 2 dense.

def trainModel():
    data_img,data_lable = loadData()
    X_train, X_test, y_train, y_test = trainTestSplit(data_img,data_lable)
    y_train =np.array(y_train)
    y_test =np.array(y_test)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train
    history = model.fit(X_train, y_train, epochs=10)

    # test
    test_loss, test_acc = model.evaluate(X_test,y_test, verbose=2)
    print(test_acc)

trainModel()
    #output result