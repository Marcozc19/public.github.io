import openpyxl
import xlwt
import xlrd
import csv
import numpy as np
import pandas as pd
from operator import itemgetter
from geopy.distance import geodesic
import NewRouteOptimizer


cwpair_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx'
cwpairwb = openpyxl.load_workbook(cwpair_path)

distancealllist = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = '表2点位距离和时间表'))
distinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表2点位距离和时间表',index_col=0))
carinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表4收运车辆表',index_col=0))
carinfolist = carinfo.values.tolist()
carlist0 = carinfolist[0:-2]
carlist1 = carinfolist[-2:]
facilityinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表3设施表',index_col=0))
demandinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表1点位表',index_col=0))


park = facilityinfo.iloc[4].values.tolist()
park = park[1:]
parkcode = park[0]
burner = facilityinfo.iloc[0].values.tolist()
burner = burner[1:]
burnercode = burner[0]

demandinfolist = demandinfo.values.tolist()

morning0 = []
morning1 = []
afternoon0 = []
afternoon1 = []

for demand in demandinfolist:
    if demand[7] == "5:00-9:00":
        if demand[-1] == 0:
            morning0.append(demand)
        else: morning1.append(demand)
    else:
        if demand[-1] == 0:
            afternoon0.append(demand)
        else: afternoon1.append(demand)

def getgeodistance(point, center):
    return geodesic((point[4], point[3]), (center[1], center[0])).km

def findmin(pointlist, center):
    mindist = 100000
    minindex = 0
    for i in range(0, len(pointlist)):
        dist = getgeodistance(pointlist[i], center)
        if dist<mindist:
            mindist = dist
            minindex = i
    return minindex

def findinsertinterval(pointname, routes):
    min = 100000
    minindex = 0
    speed = 15
    newdistfromprev = 0
    bestdistfromprev = 0
    newdisttoafter = 0
    bestdisttoafter = 0
    originaldist = 0
    bestoriginaldist = 0


    if len(routes) < 3:
        speed = 40
        return minindex, newdistfromprev / speed, newdisttoafter / speed, originaldist / speed, speed

    for i in range(0, len(routes)-1):
        newdistfrominfo = distancealllist[distancealllist['起点编码'].isin([routes[i][0]]) & (distancealllist['终点编码'].isin([pointname]))].values.tolist()[0]
        newdistfromprev = newdistfrominfo[-2] / 1000
        newdisttoinfo = distancealllist[distancealllist['起点编码'].isin([pointname]) & (distancealllist['终点编码'].isin([routes[i+1][0]]))].values.tolist()[0]
        newdisttoafter = newdisttoinfo[-2] / 1000
        newdist = newdisttoafter+newdistfromprev
        originaldistinfo = distancealllist[distancealllist['起点编码'].isin([routes[i][0]]) & (distancealllist['终点编码'].isin([routes[i+1][0]]))].values.tolist()[0]
        originaldist = originaldistinfo[-2] / 1000
        delta = newdist - originaldist
        if delta < min:
            min = delta
            minindex = i
            bestdisttoafter = newdisttoafter
            bestdistfromprev = newdistfromprev
            bestoriginaldist = originaldist
    if routes[minindex][0]== 'YQEQ-004':
        bestdistfromprev = 0
        bestoriginaldist = 0
    if routes[minindex+1][0]=='YQEQ-001':
        bestdisttoafter = 0
        bestoriginaldist = 0
    return minindex, bestdistfromprev/speed, bestdisttoafter/speed, bestoriginaldist/speed, speed


def getinitialroute(pointslist, carlist, slot):
    dummypointlist = pointslist.copy() ##dummy点位list，会将被插入线路的点位删除
    pointlen = 0
    routedict = {}
    routecount = 0
    pointcount = 0
    if slot == 'morning':
        timeconst = 4
    else:
        timeconst = 3

    for cars in carlist:
        print(cars)
        dummyfindpointlist = dummypointlist.copy() ##dummydummylist, 在查找最近点位时，会删除那些已被尝试插入但未被插入的点位
        print("dummpointlist:", len(dummypointlist))
        carplate = cars[0]
        weightconst = cars[2]
        routeinfo = [park, burner]
        center = [park[3],park[4]]
        totalcenter = [0,0]
        #totalcenter = center.copy()
        totaldist = 0
        totalweight = 0
        totaltime = 0

        while pointcount < len(pointslist):
            index = findmin(dummyfindpointlist, center)
            print(len(dummyfindpointlist))
            print("route length:", len(routeinfo))
            point = dummyfindpointlist[index]
            dummyfindpointlist.pop(index)
            pointcode = point[0]
            pointcorrdinate = [point[3], point[4]]
            pointweight = point[5]
            pointtime = point[6]
            if totalweight + pointweight < weightconst:
                insertindex, timefromprev, timetoafter, originaltime, speed = findinsertinterval(pointcode, routeinfo)
                newtime = totaltime + timefromprev + timetoafter + pointtime/60 - originaltime
                if (newtime < timeconst):
                    routeinfo.insert(insertindex+1, point)
                    totalweight += pointweight
                    totaltime = newtime
                    dummypointlist.pop(index)
                    pointcount += 1
                    '''
                    if len(routeinfo)==3:
                        totalcenter = pointcorrdinate.copy()
                        center = pointcorrdinate.copy()
                    else:
                        totalcenter[0] += pointcorrdinate[0]
                        totalcenter[1] += pointcorrdinate[1]
                        center[0] = totalcenter[0] / (len(routeinfo)-2)
                        center[1] = totalcenter[1] / (len(routeinfo)-2)
                        '''
                    totalcenter[0] += pointcorrdinate[0]
                    totalcenter[1] += pointcorrdinate[1]
                    center[0] = totalcenter[0] / (len(routeinfo) - 1)
                    center[1] = totalcenter[1] / (len(routeinfo) - 1)
                else: break
            else: break
        print(['点位数量', len(routeinfo), carplate, '现在载重', totalweight, '载重限制', weightconst, '现在时间', totaltime,'时间限制', timeconst])
        routedict[routecount] = [['点位数量', len(routeinfo), carplate, '现在载重', totalweight, '载重限制', weightconst, '现在时间', totaltime,'时间限制', timeconst], routeinfo]
        routecount += 1
    return routedict

def todataframe(routedict):
    data = pd.DataFrame(columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])
    for routecount in routedict:
        carinfo = routedict[routecount][0]
        routeinfo = routedict[routecount][1]
        routeinfo.append(carinfo)
        data = pd.concat([data, pd.DataFrame(routeinfo, columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])], axis = 0)
    return data



todataframe(NewRouteOptimizer.optimize(getinitialroute(afternoon0, carlist0, 'afternoon'))).to_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/带紧凑性的新初始解/output_test_afternoon0.xlsx')


