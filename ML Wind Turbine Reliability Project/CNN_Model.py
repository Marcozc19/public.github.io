# -*- coding: utf-8 -*-
"""
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
import math
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




#读入数据，贴上对应的标签并生成npy文件
total_label = pd.read_csv(r'')#储存label的文件地址
total_data = pd.read_csv(r'')#读取第一个数据文件，对数组初始化

splitpoint = 0 #风机分叉行数

def getlabel(name, df, i): #name:excel名称， df：写入label的文件名，i:label中开始寻找name的行数
    global splitpoint
    label = 0 
    while total_label.loc[i,'file_name'] != name:
        i+=1
    label = total_label.loc[i,'ret']
    df['label']= label
    if i>splitpoint:
        splitpoint = i

splitpoint = 0
Output = []
tempstart = 0

for part in os.listdir(r''):#文件夹目录
    domain = os.path.abspath(r'')#读取文件夹名称
    upperlayer = os.path.join(domain,part)
    for filenum in os.listdir(upperlayer):#excel文件夹目录
        domain=os.path.join(upperlayer, filenum)#文件名称
        startpoint = splitpoint
        count = 0 #同一文件夹中excel数量
        tempstart = splitpoint
        for info in os.listdir(domain):#excel文件夹
            name = os.path.join(domain, info)
            if count == 0:
                data = pd.read_csv(name)
                getlabel(info, data, tempstart)
                data = np.array(data)
                Output=data
            else:
                data = pd.read_csv(name)
                getlabel(info, data, tempstart)
                data = np.array(data)
                Output = np.concatenate((Output, data), axis = 0)
            count+=1
        np.save(domain, Output) #将文件按照文件夹的名称存储
        Output=[] #将Output置零继续读取下一个文件夹




#将无效信息（全零行）删去        
deletecount = 0

Total = []
count = 0

for part in os.listdir(r''): #输入存储npy文件的地址
    domain = os.path.abspath(r'') #同上，记录此文件目录
    savepath = os.path.abspath(r'') #写入新生成文件的存储地址
    savepath=os.path.join(savepath, part) #生成存储地址
    path = os.path.join(domain,part)
    file = np.load(path)
    deletion = []#需要删除行数
    for iter in range(len(file) - 1):
        count = 0
        i = 0
        while i < 75:
            if file[iter, i] == 0:
                count += 1
            i += 1
        if count > 74: #删除全0
            deletion.append(iter)
            deletecount += 1
    Output = np.delete(file, deletion, axis=0)
    print(deletecount)
    np.save(savepath, Output)
    
    
    
    
    
#进行相关性分析，并且缩减文件规模（考虑到数据量、运行时间与内存的限制，只采用了一个文件夹中的数据进行检验。通过大概50万条数据得到结论，可以认为比较合理）
#读取12号风机经过筛0处理的数据
test_npy = np.load('处理后数据/012.npy')

df = pd.DataFrame(test_npy)
C = df.corr()
print(C[75])

#每一列变量对label求相关系数，绝对值大于等于0.15的列留下（应该是留下了20列变量）
remain_columns = []
for i in range(C.shape[0]):
    if abs(C[75][i]) >= 0.15:
        remain_columns.append(i)
print(remain_columns)
#给初始数据按照结果降维并保存为npy文件
total_data = np.empty(shape=[0,76])
for info in os.listdir(r'D:\李畅\2021秋课程\机器学习\大作业\处理后数据'):
    domain = os.path.abspath(r'D:\李畅\2021秋课程\机器学习\大作业\处理后数据')
    info = os.path.join(domain, info)
    data = np.load(info)
    total_data = np.concatenate((total_data, data), axis = 0)

total_data = total_data[:,remain_columns]
np.save('Dimension_reduce_total_data.npy',total_data)





#对数据进行傅里叶变换降噪
select_fre = range(0,21)#21是从降维运行结果得来

def FFTvibData(vib_value):
    '''
    对处理后的振动数据进行快速傅里叶变换（FFT）
    将时域信号转变为频域信号
    返回值类型为np.array，分别为不同频率下振动的幅值和角度
    '''
    vib_fft = [np.fft.fft(vib_value[i,:]) for i in range(vib_value.shape[0])]
    vib_fft_abs = np.abs(vib_fft)
    vib_fft_ang = np.angle(vib_fft)
    
    return (vib_fft_abs, vib_fft_ang)

#筛选低频信号防止噪声干扰
def timeWindows(vib_fft):
    vib_new = []
    for i in range(vib_fft.shape[1]-1):
        a = np.max(vib_fft[:,i])
        temp = []
        for j in range(vib_fft.shape[0]):
            if vib_fft[j][i] < 0.6*a:
                temp.append(0)
            else:
                temp.append(vib_fft[j][i])
        vib_new.append(temp)
    vib_new = np.array(vib_new).T
    vib_new = np.hstack((vib_new, vib_fft[:, vib_fft.shape[1]-1].reshape(-1,1)))
    return (vib_new)

def ifftOrigin(vib_fft_abs, vib_fft_ang):
    '''
    进行快速傅里叶逆变换得到原始的时域信号
    返回值类型为np.array
    '''
    vib_fft = abs(vib_fft_abs)*np.exp(1j*vib_fft_ang)
    vib_origin = np.array([np.fft.ifft(vib_fft[i,:]).real for i in range(vib_fft.shape[0])]).reshape(-1)
    
    return (vib_origin)

def  FourierTransform(vib_gen):    
    vib_abs, vib_ang = FFTvibData(vib_gen)
    vib_abs_array = vib_abs[:,select_fre]
    vib_abs_array = timeWindows(vib_abs_array)
    vib_new = ifftOrigin(vib_abs_array, vib_ang)
    vib_new = vib_new.reshape(-1,21)
    #以下为检验效果时采取的可视化方法，不建议使用，由于代码涉及到大量图片的绘制，将大幅延长运行时间。
    '''
    time = np.array(range(vib_gen.shape[0]))
    for i in range (vib_gen.shape[1]):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes[0,0].set(title='the old data')
        axes[0,1].set(title='the old frequency')
        axes[1,0].set(title='the new frequency')
        axes[1,1].set(title='the new data')
        axes[0,0].plot(time, vib_gen[:,i])
        axes[0,1].plot(time, vib_abs[:,i])
        axes[1,0].plot(time, vib_abs_array[:,i])
        axes[1,1].plot(time, vib_new[:,i])
        plt.tight_layout()
    '''
    return(vib_new)


#读取数据并且进行切片（数据量过大直接处理会导致运行内存不够）
data = np.load(r'F:\Dimension_reduce_total_data.npy')
data1 = data[:10000000][:]
data2 = data[10000000:][:]

data1_new = FourierTransform(data1)
np.save(r'F:\data1.npy',data1_new)


data2_new = FourierTransform(data2)
np.save(r'F:\data2.npy',data2_new)    

#处理傅里叶变换产生的数据并且生成测试集
data1 = np.load('data1.npy')
data2 = np.load('data2.npy')




#读原始数据给傅里叶处理后的数据贴label
total_data = np.empty(shape=[0,1])
for info in os.listdir(r'D:\李畅\2021秋课程\机器学习\大作业\处理后数据'):
    domain = os.path.abspath(r'D:\李畅\2021秋课程\机器学习\大作业\处理后数据')
    info = os.path.join(domain, info)
    data = np.load(info)
    total_data = np.concatenate((total_data, data[:,75]), axis = 0)

old_data1 = np.concatenate((data1, data2), axis = 0)
old_data1 = old_data1[:,0:75]

old_data1 = np.concatenate((old_data1, total_data), axis = 1)

new_data1 = old_data1[0:10000000,:]
new_data2 = old_data1[10000000:20352769,:]

new_data1 = new_data1[:,remain_columns]
new_data2 = new_data2[:,remain_columns]

np.save('new_data1.npy',new_data1)
np.save('new_data2.npy',new_data2)


#处理12号风机有label的数据
new_12_train_data = test_npy[:,remain_columns]

np.save('new_12_train_data.npy', new_12_train_data)



#生成12号风机的测试集
data0 = pd.read_csv('train_labels.csv',engine='python')
data0 = np.array(data0)
domain = os.path.abspath(r'D:\李畅\2021秋课程\机器学习\大作业\data\012\012')#原始12号风机数据
total_test_data = np.empty(shape=[0,76])

for i in range(data0.shape[0]):
    
    if data0[i][2] != 1:        
        if data0[i][2] != 0:
            info = os.path.join(domain, data1[i][1])
            data_frame = pd.read_csv(info,engine='python')
            data = np.array(data_frame)
            name = np.full((data.shape[0],1),data0[i][1])
            data = np.hstack((data,name))
            total_test_data = np.vstack((total_test_data,data))

total_test_data = total_test_data[:,remain_columns]

np.save('new_12_test.npy', total_test_data)

#读取数据进行训练和预测
new_data1 = np.load('new_data1.npy')#读取经过傅里叶变换后的数据
new_data2 = np.load('new_data2.npy')

row_randarray1 = np.arange(new_data1.shape[0])
row_randarray2 = np.arange(new_data2.shape[0])
np.random.shuffle(row_randarray1)
np.random.shuffle(row_randarray2)

#从傅里叶变换总数据中随机选择1000000条数据进行训练
total_data = np.concatenate((new_data1[row_randarray1[0:500000]], new_data2[row_randarray2[0:500000]]), axis = 0)

X = np.expand_dims(total_data[:,0:20].astype(float), axis = 2)
Y = total_data[:,20]

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)  # one-hot编码

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)

#对12号风机的数据进行训练
new_12_train_data = np.load('new_12_train_data.npy')#读取12号风机已知label的数据
X1 = np.expand_dims(new_12_train_data[:,0:20].astype(float), axis = 2)
Y1 = new_12_train_data[:,20]

encoder1 = LabelEncoder()
Y1_encoded = encoder1.fit_transform(Y1)
Y1_onehot = np_utils.to_categorical(Y1_encoded)  # one-hot编码

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1_onehot, test_size=0.3, random_state=0)


# 定义神经网络
def baseline_model():
    model = Sequential()
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
    plot_model(model, to_file='model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 训练分类器
estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=1, verbose=1)  # 模型，轮数，每次数据批数，显示进度条
estimator.fit(X_train, Y_train)  # 训练模型
estimator.fit(X1_train, Y1_train)





# 将其模型转换为json
model_json = estimator.model.to_json()
with open(r"model.json", 'w')as json_file:
    json_file.write(model_json)  # 权重不在json中,只保存网络结构
estimator.model.save_weights('model.h6')



# 加载模型用做预测
json_file = open(r"model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h6")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))

print("The accuracy of the 12_train model:")
scores1 = loaded_model.evaluate(X1_test, Y1_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores1[1] * 100))

# 输出预测类别
#predicted = loaded_model.predict(X)  # 返回对应概率值
#print('predicted\n')
#print(predicted)
##predicted_label = loaded_model.predict_classes(X)  # 返回对应概率最高的标签
#predicted_label = np.argmax(loaded_model.predict(X), axis=-1)
#print("\npredicted label: " + str(predicted_label))
#print(11111111111111111111111111111111111)
# # 显示混淆矩阵
#plot_confuse(estimator.model, X_test, Y_test)  # 模型，测试集，测试集标签
#
#C = confusion_matrix(y_true=Y_test.argmax(axis=-1), y_pred=predicted_label)
#print(C)
# 可视化卷积层
#visual(estimator.model, X_train, 1)

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



