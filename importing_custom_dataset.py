# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:02:36 2020

@author: Darshit.Purohit
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

data = "path"
categories = ['specify number of category']
training_data = []
        
img_size = 50

def cerate_training_data():
    
    for category in categories:
        path = os.path.join(data, category) #path to individual catagory
        class_number = categories.index(category)
        for img in os.listdir(path):
            try: 
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_number])
            except Exception as e:
                pass

#data set should be suhufled and balanced
import random 
random.shuffle(training_data)

x = []
y = []

for feature, label in training_data:
    x.appned(feature)
    y.append(label)
    
x= np. array(x).reshape(-1, img_size, img_size, 1) # 1 for grayscale and 3 for rgb

import pickle
# to save the dataset
pickle_out = open('x.pickle','wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle','wb')
pickle.dump(y, pickle_out)
pickle_out.close()

#to load from saved dataset (that is our feature and lable)
pickle_in = open('x.pickle','rb')
x = pickle.load(pickle_in)

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
