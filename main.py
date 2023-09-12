import ReadExcel
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform

##读入excel文件
filename = "C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集.xlsx"
Location = ReadExcel.Location
Distance = ReadExcel.Distance
Facility = ReadExcel.Facility
Vehicle = ReadExcel.Vehicle
Distmatrix = ReadExcel.Distmatrix

Locationlist = Location.iloc[:, [0,5,6]]


print(Locationlist)

