# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:33:07 2020

@author: Darshit.Purohit
"""

import tensorflow as tf

#tf.__version__ # to know the version
# 28x28 image of hand writen digit 0-9
mnist = tf.keras.datasets.mnist 
#load and split data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt 
#visualize
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
#print(x_train)

''' preprocessing of the data '''
	
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normailze the data(scaling it)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train.shape)
#visuallzing after scaling
print(x_train)

''' model creation [ model architecture] i have used 2 different types '''
'''
                                        one 
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#last layer should have number of classfication(10) in the dence layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #for probability distiribution use softmmax

'''
                                        two
'''
#
#model = tf.keras.models.Sequential()
''' use convolution when input size is not set in prior, Otherwise just use dense. '''
##model.add(tf.keras.layers.Convolution2D(32, 3, 3, activation=tf.nn.relu, input_shape=(1, 28, 28)))
##model.add(tf.keras.layers.Convolution2D(32, 3, 3, activation=tf.nn.relu, input_shape=(1, 28, 28)))
##model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
##model.add(tf.keras.layers.Dropout(0.25)) # to avoid overfitting
## weights from the Convolution layers must be flattened (made 1-dimensional) 
##before passing them to the fully connected Dense layer.
#model.add(tf.keras.layers.Flatten()) 
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.25))
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#u can use any optimizer(like gradient decent(back prop)) 
model.compile(optimizer= 'adam', 
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=8)

#to know the accuracy on test set
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

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
