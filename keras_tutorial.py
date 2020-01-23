# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:33:07 2020

@author: Darshit.Purohit
"""

import tensorflow as tf

tf.__version__
# 28x28 image of hand writen digit 0-9
mnist = tf.keras.datasets.mnist 
#load and split data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt 
#visualize
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
#print(x_train)

#normailze the data(scaling it)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#visuallzie after scaling
print(x_train)

#model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#last layer should have number of classfication in the dence layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #for probability distiribution use softmmax
#u can use any optimizer(like gradient decent(back prop)) 
model.compile(optimizer= 'adam', 
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
#print(val_loss, val_acc)

#save model
model.save('myfirstmodel')

#load model with the name of model
new_model = tf.keras.models.load_model('myfirstmodel')

#topredict (it always takes a list)
prediction = new_model.predict([x_test])
print(prediction) #it results in one hot array

#to visualize in human manner
import numpy as np

print(np.argmax(prediction[0]))
plt.imshow(x_test[0], cmap = plt.cm.binary)
plt.show()

print(np.argmax(prediction[1]))
plt.imshow(x_test[1])
plt.show()
