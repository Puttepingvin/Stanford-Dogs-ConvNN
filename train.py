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

print('Start')

#Prepare data loading
trains = [im[0][0] for im in scipy.io.loadmat('train_list.mat')['file_list'][1:]]
tests = [im[0][0] for im in scipy.io.loadmat('test_list.mat')['file_list'][1:]]
dognames = os.listdir('images')
random.seed(37) #Shuffle deterministically so we can save some data for validation
shuffle(tests)
random.seed()

#Data generation functions
im_width = 224
im_height = 224
def getImage(): 
	shuffle(trains)
	i=0
	for im in trains:
		if i % 1000 == 0:
			print(i)
		i += 1
		img = tf.keras.preprocessing.image.load_img('images/' + str(im))
		img = tf.keras.preprocessing.image.img_to_array(img)
		img = tf.image.resize_with_pad(img, im_height, im_width)
		img = tf.image.random_flip_left_right(img)
		img = tf.image.random_brightness(img,0.2)
		img = tf.image.random_saturation(img, 0.5, 2)

		imgdog = im.split('/')[0]
		onehot = [int(imgdog == name) for name in dognames]		
		assert(sum(onehot) == 1)

		yield (img, onehot)
def getImageTest(): 
	for im in tests:
		img = tf.keras.preprocessing.image.load_img('images/' + str(im))
		img = tf.keras.preprocessing.image.img_to_array(img)
		img = tf.image.resize_with_pad(img, im_height, im_width)

		imgdog = im.split('/')[0]
		onehot = [int(imgdog == name) for name in dognames]		
		assert(sum(onehot) == 1)

		yield (img, onehot)

ds = tf.data.Dataset.from_generator(
    lambda: getImage(), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([im_height,im_width,3], [120])
)

dstest = tf.data.Dataset.from_generator(
    lambda: getImageTest(), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([im_height,im_width,3], [120])
)

#Build datasets
train_size = len(trains)
test_size = len(tests)
train_ds = ds.take(train_size).batch(128)
test_ds = dstest.take(1000).batch(128)



#Model architecture
class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		#Pre:
		self.conv5pre = Conv2D(16, 5,strides=2, activation='relu', padding="same")
		self.maxpool33 =  MaxPooling2D((3, 3),strides=2,padding='same')

		#Inception:
		self.conv1 = Conv2D(8, 1, activation='relu')	
		self.conv5 = Conv2D(16, 5, activation='relu', padding="same")
		self.conv3 = Conv2D(16, 3, activation='relu', padding="same")
		self.maxpool3 =  MaxPooling2D((3, 3),strides=1,padding='same')

		#Post:
		self.avgpool5 =  AveragePooling2D ((5, 5))
		self.flatten = Flatten()
		self.d2 = Dense(120)
		self.softmax = Softmax()
		self.dropout = tf.keras.layers.Dropout(0.4)

	def call(self, x):
		#Pre:
		x = self.conv5pre(x)
		x = self.maxpool33(x)

		#Inception:
		a = self.conv1(x)
		b = self.conv1(x)
		b = self.conv3(x)
		c = self.conv1(x)
		c = self.conv5(x)
		d = self.maxpool3(x)
		d = self.conv1(x)
		x = tf.concat([a, b, c, d], 3)

		#Post:
		x = self.avgpool5(x)
		x = self.flatten(x)
		x = self.dropout(x)
		x = self.d2(x)
		x = self.softmax(x)
		return x
model = MyModel()


#Loss and optimization definitions
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


#Training and testing functions
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions) 


#Main loop
EPOCHS = 10
for epoch in range(EPOCHS):
	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()

	for images, labels in train_ds:
		train_step(images, labels)

	for test_images, test_labels in test_ds:
		test_step(test_images, test_labels)

	#Uncomment for final training
	#model.save('model_1')
	print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
	)

print('Done')
