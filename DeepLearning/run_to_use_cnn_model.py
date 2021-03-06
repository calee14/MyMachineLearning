# -*- coding: utf-8 -*-
"""use_cnn_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XC1lT2Z2Ter3AD2QjltGkpb7f99Ok6aI
"""

import cv2
import tensorflow as tf
import keras
CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
  IMG_SIZE = 50
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('cat.jpg')])
print(prediction)

prediction = model.predict([prepare('dog.jpg')])
print( CATEGORIES[int(prediction[0][0])])