# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:45:07 2020

@author: Darshit.Purohit
"""
''' one way to save '''
a = [1,2,3,4,5]
import pickle
filename = open('a.pickle','wb')
pickle.dump(a, filename)
filename.close()

# to load this
file = open('a.pikle') 
b = pickle.load(file, 'rb')


''' second way to save ''' 
import numpy as np
a = np.arange(5)
np.save('a',a)

#to load the data
b = np.load('a.np')

''' if you want to zip the data '''
#np.savez('ax',a)


#for zip file do 
#print(variablename.files)



