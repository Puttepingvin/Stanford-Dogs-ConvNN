import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import random
import csv
import urllib
import matplotlib.pyplot as plt
import os
import scipy.io
from random import shuffle
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Softmax, Dropout,AveragePooling2D
from tensorflow.keras import Model

#Data preparation, model load
im_width = 224
im_height = 224
model = tf.keras.models.load_model('model_1')
tests = [im[0][0] for im in scipy.io.loadmat('test_list.mat')['file_list'][1:]]
dognames = os.listdir('images')
random.seed(37)
shuffle(tests)
random.seed()

def getImageValid(): 
	for im in tests:
		img = tf.keras.preprocessing.image.load_img('images/' + str(im))
		img = tf.keras.preprocessing.image.img_to_array(img)
		img = tf.image.resize_with_pad(img, im_height, im_width)

		imgdog = im.split('/')[0]
		onehot = [int(imgdog == name) for name in dognames]		
		assert(sum(onehot) == 1)

		yield (img, onehot)
dstest = tf.data.Dataset.from_generator(
    lambda: getImageValid(), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([im_height,im_width,3], [120])
)


#Build dataset
test_ds = dstest.skip(1000).take(5000).batch(128)

loss_object = tf.keras.losses.CategoricalCrossentropy()
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

#Test function
@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)

#Execute the test
test_loss.reset_states()
test_accuracy.reset_states()
for test_images, test_labels in test_ds:
	test_step(test_images, test_labels)
print(
f'Loss: {test_loss.result()}, '
f'Accuracy: {test_accuracy.result() * 100}'
)