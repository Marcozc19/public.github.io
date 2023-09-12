# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:54:32 2023

@author: wangxiaofang
"""

## 使用folium构建网络地图
# https://pypi.org/project/folium/
import folium
import pandas as pd
import csv
import re
import time
import numpy as np
import random

# 原数据文件格式：Customers： 名称	城市	州/省	纬度	经度
# Sites：名称	地址	城市	州/省	纬度	经度

origin_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集.xlsx'  # 原始坐标文件路径
#output_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/带紧凑性的新初始解/afternoon0/output_test_afternoon0_不计算停车场.xlsx'

df_Customers = pd.read_excel(origin_path, sheet_name='表1点位表')
df_Sites = pd.read_excel(origin_path, sheet_name='表3设施表', header=0).values.tolist()
df_CustomerOrders = pd.read_excel(origin_path)
grouped_Customer = df_Customers.groupby(['点位名称'])
carlist=pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集.xlsx',sheet_name = '表4收运车辆表'))

def MapDict(Routedict):
    m = folium.Map(location=df_Customers[['纬度', '经度']].mean(),
                   zoom_start=15,
                   control_scale=True,
                   tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                   attr='default')

    colorchoice = ['black','grey','darkorange','royalblue', 'darkviolet','crimson']
    count = 0
    for routecount in Routedict:
        routeinfo = Routedict[routecount][1]
        for point in range(0, len(routeinfo)-1):
            if routeinfo[point][1]!='停车场' and  routeinfo[point][1] != '转运站' and routeinfo[point+1][1] != '转运站' and routeinfo[point+1][1] != '处置场':
                folium.CircleMarker(location=[routeinfo[point][4], routeinfo[point][3]],
                                    radius=3,
                                    color=colorchoice[count],
                                    tooltip=str(routeinfo[point][1])).add_to(m)

                folium.PolyLine(locations=[[routeinfo[point][4], routeinfo[point][3]], [routeinfo[point+1][4], routeinfo[point+1][3]]],
                                color=colorchoice[count],
                                tooltip=routeinfo[point][1] + '-' + routeinfo[point+1][1],
                                weight=4,
                                opacity=1).add_to(m)
        count += 1
    for facility in df_Sites:
        folium.CircleMarker(location=[facility[5], facility[4]],
                        radius=3,
                        color='red',
                        tooltip=str(facility[0])).add_to(m)
    m
    m.save('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/紧凑性优化结果/afternoon0/优化后.html')
    return 0

def todict(data1):
    datalist = data1.values.tolist()
    dict = {}
    count = 0
    list = []
    for i in datalist:
        if i[0]=='点位数量':
            carinfo = i
            dict[count] = []
            dict[count].append(carinfo)
            dict[count].append(list)
            count += 1
            list = []
            continue
        else:
            list.append(i)
    return dict

#afternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件0（优化前）.xlsx', index_col=0))
#afternoon1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件1（优化前）.xlsx',index_col=0))
#morning0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件0（优化前）.xlsx',index_col=0))
#morning1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件1（优化前）.xlsx',index_col=0))
#test = pd.DataFrame(pd.read_excel("C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/带紧凑性的新初始解/output_test_afternoon0.xlsx",index_col=0))

#newmorning0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/output_test_上午且条件0（优化前）.xlsx', index_col=0))
#newafternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/output_test_下午且条件0（优化前）.xlsx', index_col=0))

test = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/紧凑性优化结果/afternoon0/优化后.xlsx', index_col=0))

MapDict(todict(test))