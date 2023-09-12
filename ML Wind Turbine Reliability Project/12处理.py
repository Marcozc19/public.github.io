# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:32:13 2021

@author: Marco
"""
import os
import pandas as pd
import numpy as np

iter = 0
count = 0
deletion = []

file = np.load(r'C:\Users\Marco\Desktop\清华大学\2021秋 大三上\机器学习\大作业数据集\全部excel\012.npy')
print(file[:, -1])
while iter in range(len(file)-1):
    if (file[iter, -1] != 0 and file[iter, -1]!=1):
        deletion.append(iter)
        count +=1
    iter +=1
        
file=np.delete(file, deletion, axis =0)   
print(count)     
np.save(r'C:\Users\Marco\Desktop\清华大学\2021秋 大三上\机器学习\大作业数据集\全部excel\012.npy', file)
