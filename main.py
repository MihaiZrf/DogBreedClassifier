import os
import cv2

import numpy as np
import pandas as pd

from glob import glob
from tensorflow import keras

from sklearn.model_selection import train_test_split

def getImage(path):
    raw_img = cv2.imread(os.path.normpath(path), cv2.IMREAD_COLOR)
    raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))

    img = raw_img.astype("float32") / 255.0

    return img

def convertData(X, y):
    img = getImage(X)

    arr = [0] * 120
    arr[y] = 1

    label = np.array(arr)
    label = label.astype("int32")

    return img, label

PATH = "Data/"
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

test_path = os.path.join(PATH, "test/", "*")
train_path = os.path.join(PATH, "train/", "*")
labels_path = os.path.join(PATH, "labels.csv")

labels_df = pd.read_csv(labels_path)

classes = labels_df["breed"].unique()

breed_id = {}

for id in range(len(classes)):
    breed_id[classes[id]] = id

labels = []

for id in glob(train_path):
    image_id_raw = id.split('\\')[-1]
    image_id = image_id_raw.split('.')[0]
    breed_list = list(labels_df[labels_df["id"] == image_id]["breed"])
    breed = breed_list[0]
    labels.append(breed_id[breed])

train_X, valid_X = train_test_split(glob(train_path), train_size = 0.9, test_size = 0.1, random_state = 22)
train_y, valid_y = train_test_split(labels, train_size = 0.9, test_size = 0.1, random_state = 22)

for i in range(len(train_X)):
    train_X[i], train_y[i] = convertData(train_X[i], train_y[i])

for i in range(len(valid_X)):
    valid_X[i], valid_y[i] = convertData(valid_X[i], valid_y[i])

X_train = np.array([x for x in train_X])
y_train = np.array([x for x in train_y])

X_valid = np.array([x for x in valid_X])
y_valid = np.array([x for x in valid_y])

base_model = keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1024, activation = 'relu'),
    keras.layers.Dense(120, activation = 'softmax')
])

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

model.fit(X_train, y_train, epochs = 10, validation_data = (X_train, y_train))