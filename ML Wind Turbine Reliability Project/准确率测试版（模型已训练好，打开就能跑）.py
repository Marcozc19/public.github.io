# -*- coding: utf-8 -*-
"""
机器学习大作业-16组
成员：贾冕、庄成、李畅
平台：spyder
框架：一维卷积神经网络
    model.add(Conv1D(2, 1, input_shape=(20, 1),kernel_initializer='he_normal',padding='same'))
    model.add(Conv1D(2, 1, activation='tanh'))
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(4, 1, activation='relu'))
    model.add(Conv1D(4, 1, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(8, 1, activation='relu'))
    model.add(Conv1D(8, 1, activation='relu'))
    model.add(MaxPooling1D(2))
    
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
版本：Python 3.9.9 
      keras 2.7.0 
      tensorflow 2.7.0
"""
import os
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as pl
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
import random
import tensorflow as tf

from keras import backend as K  
K.image_data_format() == 'channels_first'

from keras import backend as K  
K.image_data_format() == 'channels_last'



# 加载模型用做预测
json_file = open(r"model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


new_12_test = np.load('new_12_test.npy')#读取测试集数据
test_data_X = new_12_test[:,0:20]
test_data_X_expand = np.expand_dims(test_data_X.astype(float), axis = 2)
#predicted1 = loaded_model.predict(test_data_X_expand)
predicted_label1 = np.argmax(loaded_model.predict(test_data_X_expand), axis=-1)#对测试集进行预测
predicted_label1 = predicted_label1.reshape(376441,1)
names = new_12_test[:,20].reshape(376441,1)

test_labels = np.hstack((names,predicted_label1))

#计算预测准确率
data0 = pd.read_csv('train_labels.csv',engine='python')
data1 = np.array(data0)

new_test_labels = np.empty(shape=[0,2])
for i in range(data1.shape[0]):
    
    if data1[i][2] != 1:        
        if data1[i][2] != 0:
            n = 0
            m = 0   
            name = data1[i][1]
            for j in range(test_labels.shape[0]):
                if test_labels[j][0] == name:
                    n = n+1
                    #print(n)
                    if test_labels[j][1] == '1':
                        m = m+1
                        #print(m)
            #print((m,n))
            a = 0 if m/n <0.5 else 1
            new_test_labels = np.vstack((new_test_labels,(name,a)))

true_labels = pd.read_csv('true_labels.csv',engine='python')
true_labels1 = np.array(true_labels)

n = 0
m = 0
for i in range(new_test_labels.shape[0]):
    name = new_test_labels[i][0]
    for j in range(true_labels1.shape[0]):
        if true_labels1[j][1] == name:
            n = n+1
            #print(n)
            if true_labels1[j][2] == int(new_test_labels[i][1]):
                m = m+1

accuracy = m/n
print("The accuracy of the test:")#最终测试集的准确度
print(accuracy)



output = pd.DataFrame(new_test_labels)
output.columns = ['f_id','label']
output.to_csv('output.csv')








