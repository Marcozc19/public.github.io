
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 10:47:42 2021

@author: Marco
"""

import os
import numpy as np

deletecount = 0

Total = []
count = 0

for part in os.listdir(r'C:\Users\Marco\Desktop\清华大学\2021秋 大三上\机器学习\大作业数据集\处理后数据'):
    domain = os.path.abspath(r'C:\Users\Marco\Desktop\清华大学\2021秋 大三上\机器学习\大作业数据集\处理后数据')
    savepath = os.path.abspath(r'C:\Users\Marco\Desktop\清华大学\2021秋 大三上\机器学习\大作业数据集\处理后数据\012')
    savepath=os.path.join(savepath, part)
    path = os.path.join(domain,part)
    file = np.load(r'C:\Users\Marco\Desktop\清华大学\2021秋 大三上\机器学习\大作业数据集\处理后数据\012.npy')
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

    