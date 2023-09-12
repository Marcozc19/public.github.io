# -*- coding: utf-8 -*-
"""
Created on Tue Apr 4 17:15:21 2023

@author: 庄成
"""
import pandas as pd
import math
import random
import folium

#afternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件0（优化前）.xlsx', index_col=0))
distmatrix = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = 'Sheet1', index_col= 0 ))
carlist=pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = '表4收运车辆表'))
#test = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/morning0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
origin_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx'
df_Customers = pd.read_excel(origin_path, sheet_name='表1点位表')

def todict(slot, data1):
    datalist = data1.values.tolist()
    dict = {}
    count = 0
    list = []
    for i in datalist:
        if i[0]=='点位数量':
            dict[count] = []
            dict[count].append(i[2:])
            dict[count].append(list)
            count += 1
            list = []
        else:
            list.append(i)
    return dict

def fitnessfunc(route):
    totaldistance = 0
    for i in range(0, len(route)-1):
        current = route[i]
        next = route[i+1]
        distance = distmatrix.loc[current[0], next[0]]
        totaldistance += distance
    return totaldistance


def traversal_search(line):
    # 随机交换生成100个个体，选择其中表现最好的返回
    i = 0  # 生成个体计数
    line_value, line_list = [], []
    while i <= 100:
        new_line = line.copy()  # 复制当前路径
        exchange_max = random.randint(1, 5)  # 随机生成交换次数,城市数量较多时增加随机数上限效果较好
        exchange_time = 0  # 当前交换次数
        while exchange_time <= exchange_max:
            pos1, pos2 = random.randint(1, len(line) - 2), random.randint(1, len(line) - 2)  # 交换点
            new_line[pos1], new_line[pos2] = new_line[pos2], new_line[pos1]  # 交换生成新路径
            exchange_time += 1  # 更新交换次数

        new_value = fitnessfunc(new_line)  # 当前路径距离
        line_list.append(new_line)
        line_value.append(new_value)
        i += 1

    return min(line_value), line_list[line_value.index(min(line_value))]  # 返回表现最好的个体

def todataframe(routedict):
    data = pd.DataFrame(columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])
    for routecount in routedict:
        carinfo = routedict[routecount][0]
        routeinfo = []
        routes = routedict[routecount][1]
        for point in routes:
            routeinfo.append(point[:10])

        routeinfo.append(['点位数量',len(routeinfo),carinfo[0], carinfo[1], carinfo[2],  carinfo[3], carinfo[4], carinfo[5],carinfo[6],carinfo[7], carinfo[8]])
        data = pd.concat([data, pd.DataFrame(routeinfo, columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])], axis = 0)
    return data


def Initiate(Routedict):
    initial = []
    Best = []
    DummyRoutedict = Routedict.copy()
    print("Optimization")
    resultdict = {}
    for routecount in range(0, len(Routedict)):
        #SA参数
        Tend = 0.1
        T = 100
        beta = 0.99
        carinfo = Routedict[routecount][0].copy()
        route = Routedict[routecount][1].copy()
        Bestroute = route
        fit = fitnessfunc(route)
        Bestfit = fit
        initial.append(fit)
        while T>= Tend:
            newfit, newroute = traversal_search(route)

            # print(random.random(),math.exp(-(new_value-value)/T))
            if newfit <= Bestfit:  # 优于最优解
                Bestfit, Bestroute = newfit, newroute.copy()  # 更新最优解
                route, fit = newroute, newfit  # 更新当前解
            elif random.random() < math.exp(-(newfit - fit ) / T):
                route, fit = newroute.copy(), newfit  # 更新当前解
            #print('当前最优值 %.1f' % (Bestfit))
            T *= beta
        #print(len(Bestroute))
        resultdict[routecount] = []
        resultdict[routecount].append(carinfo)
        resultdict[routecount].append(Bestroute)
        Best.append(Bestfit)
    print(sum(initial))
    print(sum(Best))
    return resultdict

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
            if routeinfo[point][1]!='停车场' and  routeinfo[point][1] != '转运站' and routeinfo[point+1][1] != '转运站' :
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

    m
    m.save('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化后.html')
    return 0

#test = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
#result = Initiate(todict('morning', test))
#todataframe(result).to_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化后.xlsx')
#MapDict(result)









