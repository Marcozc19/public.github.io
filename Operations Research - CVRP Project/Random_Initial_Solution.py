# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:23:01 2021
Updated on Tue 20/07/2021 by Zhaoshan Xu
@author: 辛思源
"""
import openpyxl
from scipy import stats
import xlwt
import xlrd
import csv
import numpy as np
import pandas as pd
from operator import itemgetter
from geopy.distance import geodesic

# 导入数据并转化为list
# cwpair_path = 'C://Users/Administrator/Desktop/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化/ODPairDistance-CWDistance-20210222.xlsx'
# code_path = 'C://Users/Administrator/Desktop/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化/stand-ponits-code-all_0308.xlsx'

# cwpair_path = 'F://清华X中信项目/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化/ODPairDistance-CWDistance-20210222.xlsx'
cwpair_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集原始.xlsx'

# code_path = 'F://清华X中信项目/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化/stand-ponits-code-all_0308.xlsx'
# code_path = 'F://清华X中信项目/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化_real/0715.xlsx'
# parameter_path = 'F://清华X中信项目/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化_real/parameters.xlsx'
cwpairwb = openpyxl.load_workbook(cwpair_path)
# codewb = openpyxl.load_workbook(code_path)
# allcw = pd.DataFrame()
# allcode = pd.DataFrame()
# allcwlist = pd.DataFrame()
# allcodelist = pd.DataFrame()
allcw = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表2点位距离和时间表'))
parameters = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表4收运车辆表')).values.tolist()
facility = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表3设施表')).values.tolist()
allcwlist = allcw.values.tolist()
allcode = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表1点位表'))
allcodelist = allcode.values.tolist()
# print(allcodelist)
allcodelist = sorted(allcode.values.tolist(), key=itemgetter(0))

max_load = parameters[4][4]

first_time = True

park = facility[4][1]
transport_1 = facility[1][1]
transport_2 = facility[2][1]
# def create_cw_df(cwpair_path):
#     allcw = pd.DataFrame(pd.read_excel(cwpair_path))
#
#     allcwlist = allcw.values.tolist()
#     return allcwlist
#
# def create_code_df(code_path):
#     allcode = pd.DataFrame(pd.read_excel(code_path))
#     allcodelist = allcode.values.tolist()
#     return allcodelist


# 创建距离list和点位对list
d = []
for i in range(len(allcwlist)):
    d.append(allcwlist[i][8])
odpair = []
for i in range(len(allcwlist)):
    odpair.append([allcwlist[i][1], allcwlist[i][3]])

# 筛选时间窗口和道路通行条件
order = ['序号', '垃圾量（升/处）', '作业停留时间（分钟）', '车辆通行条件']
# namoinfo = allcode.loc[(allcode['车辆通行条件'] == parameters[0][3]) & (allcode['允许时间窗口'] == parameters[0][4])]
namoinfo = allcode
# print("allcode")
# print(allcode)
namo = namoinfo[order].sort_values(by=['序号'])
namolist = namo.values.tolist()
# namoinfolist = namoinfo.values.tolist()
namoinfolist = sorted(namoinfo.values.tolist(), key=itemgetter(0))
# print("namoinfolist")
# print(namoinfolist)
# #创建code list 需求和时间list
# code = []
# for i in range(len(namoinfolist)):
#     code.append(namoinfolist[i][0])
# g = []
# for i in range(len(namolist)):
#     namolist[i][1] *= 0.2
#     g.append(namolist[i][1])
#
# t = []
# for i in range(len(namolist)):
#     t.append(namolist[i][2])
#
# namoPoint = namo['序号']
# namocodelist = namoPoint.values.tolist()
# namocw = allcw[(allcw['起点编码'].isin(namocodelist))&(allcw['终点编码'].isin(namocodelist))]
#
# df = namocw.sort_values(['drivedistance'],ascending=True)
# df1 = df.values.tolist()
#
# #创建初始线路list
# Routes = []
# for i in range(len(namocodelist)):
#     Routes.append([namocodelist[i]])#初始路线
# iRoutes = []
# for i in range(len(namocodelist)):
#     iRoutes.append([namocodelist[i]])
# 创建code list 需求和时间list
code = []
for i in range(len(allcodelist)):
    code.append(namoinfolist[i][0])
g = []
for i in range(len(namolist)):
    namolist[i][1] *= 0.2
    g.append(namolist[i][1])

t = []
for i in range(len(namolist)):
    t.append(namolist[i][2])

namoPoint = namo['序号']
namocodelist = namoPoint.values.tolist()
namocw = allcw[(allcw['起点编码'].isin(namocodelist)) & (allcw['终点编码'].isin(namocodelist))]

df = namocw.sort_values(['drivedistance'], ascending=True)
df1 = df.values.tolist()


def main_func(df, namoinfolist, first_time, namocode, odpair, namolist, d, max_load, time_remain, work_speed,
              back_speed):
    df1 = df.copy()
    print("call mainfunc")
    '''
    df1: 按照距离排序的符合时间窗以及车辆通行约束的点位：[起点名称,起点编码,终点名称,终点编码,起点经度,起点纬度,终点经度,终点纬度,drivedistance,时间(秒)]
    namoinfolist: 按照点位编号排序的dataframe：[序号，点位名称，所属路街，经度，纬度，垃圾量（升/处），作业停留时间（分钟），允许时间窗口，禁止时间窗口，收运频次（次/日），车辆通行条件]
    first_time: 是否为第一次
    namocode: 序号list
    odpair: 点位对list
    namolist: 按序号排序的dataframe：['序号', '垃圾量（升/处）', '作业停留时间（分钟）', '车辆通行条件']
    d: 距离list
    max_load: 车辆最大载重
    time_remain: 剩余作业时间
    work_speed: 作业速度
    back_speed: 运动速度
    '''

    rule1 = max_load
    rule2 = time_remain

    # 点位代码list
    code = []
    for i in range(len(namoinfolist)):
        code.append(namoinfolist[i][0])

    # 点位垃圾量
    g = []
    for i in range(len(namolist)):
        # namolist[i][1] *= 0.2
        g.append(namolist[i][1])

    # 点位作业时间
    t = []
    for i in range(len(namolist)):
        t.append(namolist[i][2])

    # 创建初始线路list
    Routes = []
    for i in range(len(namocode)):
        Routes.append([namocode[i]])  # 初始路线
    iRoutes = []
    for i in range(len(namocode)):
        iRoutes.append([namocode[i]])

    # print("before main func")
    # print(Routes)
    # 算法主程序

    for count in range(len(df1)):
        startRoute = []
        endRoute = []
        routeDemand = 0
        # 在这里加入随机性
        # 在df1中遍历连接线路，若route[j]的末尾与route[j+1]的起点相连接，则先择df1[i]的线路
        rand = stats.geom.cdf(1, 0.40)
        i = int(rand * len(df1))

        for j in range(len(Routes)):
            if (df1[i][1] == Routes[j][-1]):
                endRoute = Routes[j]
                # print("endRoute:", i, j, endRoute)
            elif (df1[i][3] == Routes[j][0]):
                startRoute = Routes[j]  # 遍历依次判断点位是否与起点终点相连
                # print("startRoute", i, j, startRoute)

            if ((len(startRoute) != 0) and (len(endRoute) != 0)):
                for k in range(len(startRoute)):
                    routeDemand += g[iRoutes.index([startRoute[k]])]
                for k in range(len(endRoute)):
                    routeDemand += g[iRoutes.index([endRoute[k]])]  # 计算当前线路总重量
                routeDistance = 0
                routet_stay = 0
                routetime = 0
                routestore = endRoute + startRoute
                for i1 in range(len(routestore) - 1):
                    routeDistance += d[odpair.index([routestore[i1], routestore[i1 + 1]])]  # 计算当前线路总里程
                for i2 in range(len(routestore)):
                    routet_stay += t[iRoutes.index([routestore[i2]])]  # 计算当前线路总停留时间
                final_trans = transport_1  # 确定中转站
                if (d[odpair.index([routestore[-1], transport_1])] > d[odpair.index([routestore[-1], transport_2])]):
                    final_trans = transport_2
                routetime = routet_stay + routeDistance / work_speed + (d[odpair.index([park, routestore[0]])] + d[
                    odpair.index([routestore[-1], final_trans])]) / back_speed
                # print(routeDistance)
                # 按照限制规则对​​路线进行更改 时间 3*60min，或者4*40
                if (routeDemand <= int(rule1)) and (routetime <= int(rule2)):
                    Routes.remove(startRoute)
                    Routes.remove(endRoute)
                    Routes.append(endRoute + startRoute)
                    df1.remove(df1[i])
                    #print(Routes)
                break
    # print("after break ")
    # print(Routes)
    # 按照指定格式进行输出

    resultlist = []  # 形成初步路线结果
    for i in range(len(Routes)):
        dis = 0
        weight = 0
        t_stay = 0
        time = 0
        distance = 0
        counts = 0
        kdis = 0
        for j in range(len(Routes[i]) - 1):
            dis += d[odpair.index([Routes[i][j], Routes[i][j + 1]])]
        for j in range(len(Routes[i])):
            weight += g[iRoutes.index([Routes[i][j]])]
            t_stay += t[iRoutes.index([Routes[i][j]])]
            counts += 1
        final_trans = transport_1
        if (d[odpair.index([Routes[i][-1], transport_1])] > d[odpair.index([Routes[i][-1], transport_2])]):
            final_trans = transport_2
        distance = dis + d[odpair.index([park, Routes[i][0]])] + d[odpair.index([Routes[i][-1], final_trans])]
        kdis = d[odpair.index([park, Routes[i][0]])] + d[odpair.index([Routes[i][-1], final_trans])]
        time = t_stay + dis / (15 * 1000 / 60) + (
                    d[odpair.index([park, Routes[i][0]])] + d[odpair.index([Routes[i][-1], final_trans])]) / (
                           40 * 1000 / 60)
        resultlist.append([park, '停车场', '', facility[0][5], facility[0][6], 0, facility[0][4], 'all', 0, 1,
                           '3吨可通行'])  # 加了一行停车场的信息
        for j in range(len(Routes[i])):
            if first_time == True:
                resultlist.append(namoinfolist[code.index(Routes[i][j])])
            else:
                resultlist.append(namoinfolist[code.index(Routes[i][j])])
        resultlist.append([final_trans, '转运站', '', facility[4][5], facility[4][6], 0, facility[4][4], 'all', 0, 1,
                           '3吨可通行'])  # 添加起点和终点
        resultlist.append(
            ['点位数量', counts, '重量', weight * 0.001, '空驶行程', kdis * 0.001, '路程', distance * 0.001, '时间',
             time * 0.016666667, '作业停留时间', t_stay * 0.016666667])
    print("end mainfunc")
    return resultlist


# 将所有线路和指标结果输出excel表格

output = []


# 寻找此时间段运行条件下的负载最大路线
def find_max(a_list):
    print("call findmax")
    max = 0

    # item : all route and route summary
    for index, item in enumerate(a_list):
        # detail of i
        for j in range(0, len(item)):
            if item[j] == '重量':
                # print(item[j+1])
                if item[j + 1] / max_load > max:
                    max_route = []
                    max = item[j + 1] / max_load

                    k = index - 1
                    max_route.insert(0, a_list[index])
                    while a_list[k][0] != '点位数量' and k >= 0:
                        max_route.insert(0, a_list[k])
                        k -= 1
    print("end findmax")
    return max_route


def binarySearch(arr, l, r, x):
    # Check base case
    if r >= l:

        mid = l + (r - l) // 2

        # If element is present at the middle itself
        if arr[mid][0] == x:
            return mid

        # If element is smaller than mid, then it
        # can only be present in left subarray
        elif arr[mid][0] > x:
            return binarySearch(arr, l, mid - 1, x)

        # Else the element can only be present
        # in right subarray
        else:
            return binarySearch(arr, mid + 1, r, x)

    else:
        # Element is not present in the array
        return -1


def output(all_code, allcode_list, statement, car_list):
    print("call output")
    completeroute = []
    # 创建距离list和点位对list
    d = []  # 距离list
    for i in range(len(allcwlist)):
        d.append(allcwlist[i][8])
    odpair = []  # 点位对list
    for i in range(len(allcwlist)):
        odpair.append([allcwlist[i][1], allcwlist[i][3]])

    # 筛选时间窗口和道路通行条件
    order = ['序号', '垃圾量（升/处）', '作业停留时间（分钟）', '车辆通行条件']
    namoinfo = all_code

    namo = namoinfo[order].sort_values(by=['序号'])  # 以序号排序的包含['序号', '垃圾量（升/处）', '作业停留时间（分钟）', '车辆通行条件']
    namolist = namo.values.tolist()

    namoinfolist = sorted(namoinfo.values.tolist(), key=itemgetter(0))

    namoPoint = namo['序号']
    namocodelist = namoPoint.values.tolist()
    namocw = allcw[(allcw['起点编码'].isin(namocodelist)) & (allcw['终点编码'].isin(namocodelist))]

    df = namocw.sort_values(by=['drivedistance', '起点纬度'], ascending=[True, True])
    df1 = df.values.tolist()

    result_list = main_func(df1, namoinfolist, statement, namocodelist, odpair, namolist, d, car_list[0][1],
                            car_list[0][2] * 60, car_list[0][3] * 1000 / 60, car_list[0][4] * 1000 / 60)
    '''
    df1: 按照距离排序的符合时间窗以及车辆通行约束的点位
    namoinfolist: 按照点位编号排序的dataframe：[序号，点位名称，所属路街，经度，纬度，垃圾量（升/处），作业停留时间（分钟），允许时间窗口，禁止时间窗口，收运频次（次/日），车辆通行条件]
    Statement: 是否为第一次
    namocodelist: 序号list
    odpair: 点位对list
    namolist: 按序号排序的dataframe：['序号', '垃圾量（升/处）', '作业停留时间（分钟）', '车辆通行条件']
    d: 距离list 
    carlist: 运收车辆list：[车牌号,"核载（kg）", 最大载重（kg）, 3.5/4, 作业速度（km/h）, 回场速度（km/h）]
    '''

    statement = False
    result11 = pd.DataFrame(result_list)
    # print("resultlist")
    # print(result_list)
    # print("list")
    # print(namoinfolist)
    result11.to_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/resultlist_test.xlsx')

    routes = []
    car_number = 0
    total_time = 0
    while len(namoinfolist) >= len(result_list[0]):
        route_selected = find_max(result_list)
        total_time += route_selected[-1][9]
        # 测试第二趟加第一躺总时间，如果总时间超出半天工作量，则不添加第二趟，为下一辆车重新执行算法
        if total_time > car_list[car_number][2]:
            print(car_list[car_number])
            car_number += 1
            total_time = 0
        else:
            # routes.append([""])
            route_selected[-1].append(car_list[car_number][0])

            routes += route_selected

            route_selected.pop()

            for i in range(1, len(route_selected) - 1):
                index = binarySearch(namoinfolist, 0, len(namoinfolist) - 1, route_selected[i][0])
                namocodelist.remove(route_selected[i][0])
                namoinfolist.pop(index)

            mark_list = []
            # 将已选择的需求点从距离list中删除
            for i in df1:
                for j in route_selected:

                    if i[1] == j[0] or i[3] == j[0]:

                        if i in df1:
                            df1.remove(i)
        # namo.pop(0)
        try:
            if len(namoinfolist) > 0:
                result_list = main_func(df1, namoinfolist, statement, namocodelist, odpair, namolist, d,
                                        car_list[car_number][1], car_list[car_number][2] * 60,
                                        car_list[car_number][3] * 1000 / 60, car_list[car_number][4] * 1000 / 60)
        except IndexError:
            pass

    print("end output")
    print("routes:", routes)
    return routes


# print("1")
# print(output(df1, namoinfolist,allcodelist,first_time,namocodelist))


'''
starting program
'''

rows = allcode.shape[0]
time_limit_list = []
morning0, morning1, afternoon0, afternoon1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
# morningroute0, morningroute1, afternoonroute0, afternoonroute1 = []

for i in range(rows):
    temp = allcode["允许时间窗口"][i]
    if temp not in time_limit_list:
        time_limit_list.append(temp)  # 将部门的分类存在一个列表中

# allcodelist = allcode.values.tolist()
for time_limit in time_limit_list:
    new_df = pd.DataFrame()
    for i in range(0, rows):
        if allcode["允许时间窗口"][i] == time_limit:
            new_df = pd.concat([new_df, allcode.iloc[[i], :]], axis=0, ignore_index=True)
    if time_limit == "14:00-17:00":  # 分开上下午需求点
        vehicle_limit_list = []
        inside_rows = new_df.shape[0]
        for i in range(inside_rows):
            temp = new_df["车辆通行条件"][i]
            if temp not in vehicle_limit_list:
                vehicle_limit_list.append(temp)
        for vehicle_limit in vehicle_limit_list:
            another_df = pd.DataFrame()
            for i in range(0, inside_rows):
                if new_df["车辆通行条件"][i] == vehicle_limit:
                    another_df = pd.concat([another_df, new_df.iloc[[i], :]], axis=0, ignore_index=True)
            if vehicle_limit == 0:
                # another_df.to_excel("下午且条件0.xlsx")
                codelist = another_df.values.tolist()
                carlist = []
                for i in range(0, len(parameters) - 2):
                    row = []
                    row.append(parameters[i][1])
                    row.append(parameters[i][3])
                    row.append(3.5)
                    row.append(parameters[i][-3])
                    row.append(parameters[i][-1])
                    carlist.append(row)

                result = pd.DataFrame(output(another_df, codelist, first_time, carlist),
                                      columns=['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）',
                                               '作业停留时间（分钟）', '允许时间窗口', '禁止时间窗口', '收运频次（次/日）',
                                               '车辆通行条件', '', ''])
                result.to_excel(
                    'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件0（随机优化前）.xlsx')
            else:
                # another_df.to_excel("下午且条件1.xlsx")
                codelist = another_df.values.tolist()
                carlist = []
                for i in range(-2, len(parameters)):  ###不懂
                    row = []
                    row.append(parameters[i][1])
                    row.append(parameters[i][3])
                    row.append(3.5)
                    row.append(parameters[i][-3])
                    row.append(parameters[i][-1])
                    carlist.append(row)
                result = pd.DataFrame(output(another_df, codelist, first_time, carlist),
                                      columns=['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）',
                                               '作业停留时间（分钟）', '允许时间窗口', '禁止时间窗口', '收运频次（次/日）',
                                               '车辆通行条件', '', ''])
                result.to_excel(
                    'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件1（随机优化前）.xlsx')
    else:  # 上午的需求点
        vehicle_limit_list = []
        inside_rows = new_df.shape[0]
        # 创建车辆通行条件
        for i in range(inside_rows):
            temp = new_df["车辆通行条件"][i]
            if temp not in vehicle_limit_list:
                vehicle_limit_list.append(temp)

        for vehicle_limit in vehicle_limit_list:
            another_df = pd.DataFrame()
            for i in range(0, inside_rows):
                if new_df["车辆通行条件"][i] == vehicle_limit:
                    another_df = pd.concat([another_df, new_df.iloc[[i], :]], axis=0, ignore_index=True)
            # 车辆通行条件=0
            if vehicle_limit == 0:
                # another_df.to_excel("上午且条件0.xlsx")
                codelist = another_df.values.tolist()
                carlist = []
                for i in range(0, len(parameters) - 2):
                    row = []
                    row.append(parameters[i][1])
                    row.append(parameters[i][3])
                    row.append(4)
                    row.append(parameters[i][-3])
                    row.append(parameters[i][-1])
                    carlist.append(row)
                result = pd.DataFrame(output(another_df, codelist, first_time, carlist),
                                      columns=['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）',
                                               '作业停留时间（分钟）', '允许时间窗口', '禁止时间窗口', '收运频次（次/日）',
                                               '车辆通行条件', '', ''])
                result.to_excel(
                    'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件0（随机优化前）.xlsx')
            # 车辆通行条件=1
            else:
                # another_df.to_excel("上午且条件1.xlsx")
                codelist = another_df.values.tolist()
                carlist = []
                for i in range(-2, len(parameters)):
                    row = []
                    row.append(parameters[i][1])
                    row.append(parameters[i][3])
                    row.append(4)
                    row.append(parameters[i][-3])
                    row.append(parameters[i][-1])
                    carlist.append(row)
                result = pd.DataFrame(output(another_df, codelist, first_time, carlist),
                                      columns=['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）',
                                               '作业停留时间（分钟）', '允许时间窗口', '禁止时间窗口', '收运频次（次/日）',
                                               '车辆通行条件', '', ''])
                result.to_excel(
                    'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件1（随机优化前）.xlsx')

# result = pd.DataFrame(output(allcode,allcodelist,first_time), columns=['点位编号', '点位名称', '所属路街','桶型号（L）','桶数/个','经度','纬度','是否坡路','半桶数/个','满桶数/个','空桶数/个','垃圾量（升/处）','作业停留时间（分钟）','允许时间窗口','禁止时间窗口','收运频次（次/日）','车辆通行条件'])
# result.to_excel('F://清华X中信项目/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化_real/output_test.xlsx')
# resul = pd.DataFrame(df1)
# resul.to_excel('C://Users/user/Desktop/清华X中信项目/清华X中信项目/中信智慧环卫项目/智慧环卫项目资料/垃圾收运线路优化/df1.xlsx')
# 后续需要选择一条满载率最高的线路，然后删除这条线路上的点。更换其它车辆载重，再运行主程序

