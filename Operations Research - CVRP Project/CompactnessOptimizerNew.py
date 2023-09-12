# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:31:01 2023

@author: 庄成
"""

import pandas as pd
from geopy.distance import geodesic
import folium
import InterRouteOptimizer
import os
import random


afternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/初始解/output_test_下午且条件0（优化前）.xlsx', index_col=0))
#afternoon1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件1（优化前）.xlsx',index_col=0))
morning0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/初始解/output_test_上午且条件0（优化前）.xlsx',index_col=0))
#morning1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件1（优化前）.xlsx',index_col=0))
carlist=pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集.xlsx',sheet_name = '表4收运车辆表'))
distancealllist = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = '表2点位距离和时间表'))
origin_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx'
df_Customers = pd.read_excel(origin_path, sheet_name='表1点位表')

'''
将dataframe转换成dict
'''
def todict(slot, data1):
    datalist = data1.values.tolist()
    dict = {}
    count = 0
    list = []
    for i in datalist:
        if i[0]=='点位数量':
            carplate = i[-1]
            weightconst = carlist[carlist['车牌号'].isin([carplate])].values.tolist()[0][2]
            if slot == 'morning': timeconst = 4
            else: timeconst = 3
            carinfo = [i[-1], '现有载重',i[3]*1000, weightconst, '现在耗时', i[9], timeconst]
            dict[count] = []
            dict[count].append(carinfo)
            dict[count].append(list)
            count += 1
            list = []
        else:
            list.append(i)
    return dict

'''
寻找给定路线的中心点
'''
def getcenter(routedict):
    center = []
    for routecount in range(0, len(routedict)):
        carinfo = routedict[routecount][0]
        routeinfo = routedict[routecount][1]
        longitude = 0
        latitude = 0
        i = 0
        for count in range(0, len(routeinfo)-1):
            if routeinfo[count][1] == '停车场' or routeinfo[count][1] == '转运站':
                continue
            else:
                longitude += float(routeinfo[count][3])
                latitude += float(routeinfo[count][4])
                i+=1
        longitudebar = longitude / i
        latitudebar = latitude / i
        center.append([longitudebar, latitudebar])
    return center

'''
第二种中点选择，其返回值为线路的中间点
'''
def getroutecenter(routedict):
    center = []
    for routecount in range(0, len(routedict)):
        routeinfo = routedict[routecount][1]
        length = len(routeinfo)
        mid = int(length/2)
        midpoint =  routeinfo[mid]
        center.append([midpoint[3], midpoint[4]])
    return center

'''
将距离其他中点更近的路线寻找出来
'''
def getpoints(centerlist, routedict):
    RemainingRoutedict = {}  # 在遍历线路时将所有线路提取出来
    Exchangepointdict = []
    remaininglist = []
    comp = [0]  # 记录路线点位到中心点总距离
    compd = [0]  # 记录COMPd
    count = 0
    for routenum in range(0, len(routedict)):
        carinfo = routedict[routenum][0]
        routeinfo = routedict[routenum][1]
        for point in routeinfo:
            #point = routeinfo[i] #当前点位
            if point[1] == '停车场' or point[1]  == '转运站':
                remaininglist.append(point)
                continue
            else:
                distancelist = []
                count += 1
                for j in centerlist:
                    distancelist.append(geodesic((point[4], point[3]), (j[1], j[0])).km)
                comp[routenum] += float(distancelist[routenum])
                closetroute = distancelist.index(min(distancelist))
                if  closetroute != routenum:
                    Exchangepointdict.append(point[0:12])
                else:
                    remaininglist.append(point)
        RemainingRoutedict[routenum] = []
        RemainingRoutedict[routenum].append(carinfo)
        RemainingRoutedict[routenum].append(remaininglist)
        remaininglist = []
        compd[routenum] = comp[routenum] / count
        comp.append(0)
        compd.append(0)
        count = 0
    return RemainingRoutedict, Exchangepointdict, comp, compd

'''
将需要被插入的点位进行排序
'''
def Sortdist(centerlist, pointlist):
    Exchangepointdict = {}
    for point in pointlist:
        distancelist = []
        for j in centerlist:
            distancelist.append(geodesic((point[4], point[3]), (j[1], j[0])).km)
        point.append(min(distancelist))
        if (distancelist.index(min(distancelist)) not in Exchangepointdict.keys()):
            Exchangepointdict[distancelist.index(min(distancelist))] = []
            Exchangepointdict[distancelist.index(min(distancelist))].append(point)
        else:
            Exchangepointdict[distancelist.index(min(distancelist))].append(point)

    def sortlist(element):
        return element[-1]

    for num in range(0, len(Exchangepointdict)):
        if (num not in Exchangepointdict.keys()):
            Exchangepointdict[num] = []
            continue
        alist = Exchangepointdict[num]
        alist.sort(key=sortlist, reverse=True)
        Exchangepointdict[num] = alist
    return Exchangepointdict

def getcurrentinfo(routedict):
    routeinfobook = {}
    for routecount in range(0, len(routedict)):
        carinfo = routedict[routecount][0]
        routeinfo = routedict[routecount][1]
        routelist = []
        currentdemand = 0
        currenttime = 0
        prev = []
        for i in routeinfo:
            if i[1] == '停车场':
                prev = i
                routelist.append(i)
                continue
            if prev[1] == '停车场':
                prev = i
                i.append(0)
                i.append(0)
                routelist.append(i)
                continue
            currentdemand += i[5]#计算当前载重
            pairdistinfo = distancealllist[(distancealllist['起点编码'].isin([prev[0]]))&(distancealllist['终点编码'].isin([i[0]]))].values.tolist()
            disttoprev = pairdistinfo[0][-2]/1000
            timetoprev = disttoprev/15
            currenttime += (i[6] / 60 + timetoprev)
            prev = i
            i.extend([disttoprev, timetoprev])
            routelist.append(i)
        carinfo[2] = currentdemand
        carinfo[5] = currenttime
        routeinfobook[routecount] = []
        routeinfobook[routecount].append(carinfo)
        routeinfobook[routecount].append(routelist)
    return routeinfobook

def getcurrentinfo2(routedict):
    routeinfobook = {}
    for routecount in range(0, len(routedict)):
        carinfo = routedict[routecount][0]
        routeinfo = routedict[routecount][1]
        routelist = []
        currentdemand = 0
        currenttime = 0
        prev = []
        for i in routeinfo:
            if i[1] == '停车场':
                prev = i
                routelist.append(i)
                continue
            if prev[1] == '停车场':
                prev = i
                i.append(0)
                i.append(0)
                routelist.append(i)
                continue
            if i[1] == '转运站':
                routelist.append(i)
                continue
            currentdemand += i[5]#计算当前载重
            pairdistinfo = distancealllist[(distancealllist['起点编码'].isin([prev[0]]))&(distancealllist['终点编码'].isin([i[0]]))].values.tolist()
            disttoprev = pairdistinfo[0][-2]/1000
            timetoprev = disttoprev/15
            currenttime += (i[6] / 60 + timetoprev)
            prev = i
            i[-1] = timetoprev
            i[-2] = disttoprev
            routelist.append(i)
        carinfo[2] = currentdemand
        carinfo[5] = currenttime
        routeinfobook[routecount] = []
        routeinfobook[routecount].append(carinfo)
        routeinfobook[routecount].append(routelist)
    return routeinfobook

def findinsertinterval(pointname, routes):
    min = 100000
    best = []
    minindex = 0
    speed = 15
    i = 0
    newdistfromprev = 0
    bestdistfromprev = 0
    newdisttoafter = 0
    bestdisttoafter = 0
    originaldist = 0
    for i in range(0, len(routes)-2):
        newdistfrominfo = distancealllist[distancealllist['起点编码'].isin([routes[i][0]]) & (distancealllist['终点编码'].isin([pointname]))].values.tolist()[0]
        newdistfromprev = newdistfrominfo[-2]/1000
        newdisttoinfo = distancealllist[distancealllist['起点编码'].isin([pointname]) & (distancealllist['终点编码'].isin([routes[i+1][0]]))].values.tolist()[0]
        newdisttoafter = newdisttoinfo[-2] / 1000
        newdist = newdisttoafter+newdistfromprev
        originaldist = routes[i+1][-2]
        delta = newdist - originaldist
        if delta < min:
            min = delta
            minindex = i
            bestdistfromprev = newdistfromprev
            bestdisttoafter = newdisttoafter
    if routes[minindex][0]== 'YQEQ-004':
        bestdistfromprev = 0
        speed = 40
    return minindex, bestdistfromprev/speed, bestdisttoafter/speed, originaldist/speed, speed


def findinsertinterval2(point, routeinfo):
    pointlong = point[3]
    pointlat = point[4]
    minindex = 0
    mindist = 10000
    for i in range(0, len(routeinfo)):
        distance = (geodesic((pointlong, pointlat), (routeinfo[i][3], routeinfo[i][4])).km)
        if distance<mindist:
            mindist = distance
            minindex = i
    return minindex


def InsertPoints(RouteDict, InerstPointDict):
    pointsleft = []
    NewRouteDict = RouteDict
    for routecount in range(0, len(RouteDict)):
        carinfo = RouteDict[routecount][0]
        routeinfo = RouteDict[routecount][1]
        weightconst = carinfo [3]
        timeconst = carinfo[-1]
        if routecount in InerstPointDict.keys():
            for points in InerstPointDict[routecount]:
                totaldemand = carinfo[2]
                totaltime = carinfo[-2]
                if (totaldemand + points[5]) < weightconst:
                    insertindex, timefromprev, timetoafter, originaltime, speed = findinsertinterval(points[0], routeinfo)
                    newtime = totaltime + timefromprev + timetoafter + points[6]/60 - originaltime
                    if (newtime < timeconst):
                        insertinfo = points[:-1]
                        insertinfo.append(timefromprev*speed)
                        insertinfo.append(timefromprev)
                        routeinfo.insert(insertindex+1, insertinfo)
                        routeinfo[insertindex+2][-1] = timetoafter
                        routeinfo[insertindex+2][-2] = timetoafter*speed
                        carinfo[2] = totaldemand + points[5] #更新线路需求
                        carinfo[-2] = newtime #更新线路时间
                        if (totaldemand + points[5]> weightconst):
                            os.system("pause")
                    else:
                        pointsleft.append(points)
                        #print("cannot insert into route due to time")
                else:
                    pointsleft.append(points)
                    #print("cannot insert into route due to weight")
        else:
            continue
        NewRouteDict[routecount][0] = carinfo
        NewRouteDict[routecount][1] = routeinfo

    return pointsleft, RouteDict

def InsertLeftover(Routedict, Pointsleft):
    leftpointslist = []

    dummyleftover = Pointsleft

    def sortlist(element):
        return element[5]

    alist = dummyleftover
    alist.sort(key=sortlist, reverse=True)
    sortedleftover = alist

    for points in sortedleftover:
        BestInsertRoute = 100
        BestInsertIndex = 100
        Bestnewtime = 0
        mintimeincrease = 10000
        for routecount in range(0, len(Routedict)):  # 寻找每个点到每个线路中心的距离
            carinfo = Routedict[routecount][0]
            routeinfo = Routedict[routecount][1]
            weightconst = carinfo[3]
            timeconst = carinfo[-1]
            totalweight = carinfo[2]
            totaltime = carinfo[-2]
            if (points[5] + totalweight) < weightconst:
                insertindex, timefromprev, timetoafter, originaltime, speed = findinsertinterval(points[0], routeinfo)
                newtime = totaltime + timefromprev + timetoafter + points[6] / 60 - originaltime
                if newtime<timeconst:
                    if timefromprev + timetoafter- originaltime < mintimeincrease:
                        mintimeincrease = timefromprev + timetoafter- originaltime
                        BestInsertRoute = routecount
                        BestInsertIndex = insertindex
                        Bestnewtime = newtime
        if BestInsertRoute != 100:
            routeinfo = Routedict[BestInsertRoute][1]
            carinfo = Routedict[BestInsertRoute][0]
            totalweight = carinfo[2]
            insertinfo = points
            insertinfo.append(timefromprev*speed)
            insertinfo.append(timefromprev)
            routeinfo.insert(BestInsertIndex+1, insertinfo)
            routeinfo[BestInsertIndex+2][-1] = timetoafter
            routeinfo[BestInsertIndex+2][-2] = timetoafter*speed
            carinfo[2] = totalweight + points[5] #更新线路需求
            carinfo[-2] = Bestnewtime #更新线路时间
            Routedict[BestInsertRoute][1] = routeinfo
            Routedict[BestInsertRoute][0] = carinfo
            #print(points[1], "insert into route", BestInsertRoute, "at index", BestInsertIndex)
        else: leftpointslist.append(points)
    return leftpointslist, Routedict


def MapDict(Routedict, leftover, center, integer):
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

    for point in leftover:
        folium.CircleMarker(location=[point[4], point[3]],
                            radius=3,
                            color='red',
                            tooltip=str(point[1])).add_to(m)

    for point in center:
        folium.CircleMarker(location=[point[1], point[0]],
                            radius=3,
                            color='green',
                            tooltip=str(point[1])).add_to(m)

    m
    m.save('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Map/测试%s.html'%integer)
    return 0

def todataframe(routedict):
    data = pd.DataFrame(columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])
    for routecount in routedict:
        carinfo = routedict[routecount][0]
        routeinfo = []
        routes = routedict[routecount][1]
        for point in routes:
            routeinfo.append(point[:10])

        routeinfo.append(['点位数量',len(routeinfo),carinfo[0], carinfo[1], carinfo[2], '载重限制', carinfo[3], carinfo[4], carinfo[5], '现在时间',carinfo[6]])
        data = pd.concat([data, pd.DataFrame(routeinfo, columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])], axis = 0)
    return data


def getlength(routedict):
    length = 0
    for routecount in routedict:
        length+=len(routedict[routecount][1])
    return length

def calculatecompd(Routedict):
    compdlist = []
    center = getcenter(Routedict)
    for routecount in range(0, len(Routedict)):
        totaldist = 0
        route = Routedict[routecount]
        carinfo = route[0]
        routeinfo = route[1]
        for point in routeinfo:
            if point[1] != '停车场' and point[1]  != '转运站':
                totaldist += (geodesic((point[4], point[3]), (center[routecount][1], center[routecount][0])).km)
        compdlist.append(totaldist/len(routeinfo))
    return compdlist


def Initiate(slot, rawdata):
    centercompdlist = []
    switchcount = 0
    switchnum = 0
    compdlist = []
    print(len(rawdata))
    InitialRouteDict = todict(slot, rawdata)
    centercompdlist.append(sum(calculatecompd(InitialRouteDict)))
    print(getlength(InitialRouteDict))
    initialcenter = getroutecenter(InitialRouteDict)
    RemainingRouteDict, Pointsswitchlist, comp, compd = getpoints(initialcenter, InitialRouteDict)
    compdlist.append(sum(compd))
    print(getlength(RemainingRouteDict),len(Pointsswitchlist), getlength(RemainingRouteDict)+len(Pointsswitchlist))
    #MapDict(RemainingRouteDict, Pointsswitchlist, initialcenter)
    #Newcenter = getroutecenter(RemainingRouteDict)
    PointswitchDict = Sortdist(initialcenter, Pointsswitchlist)
    CurrentRouteDict = getcurrentinfo2(RemainingRouteDict) ###'''返回的值中一行数据的倒数第二为到前点的距离，倒数第一为到前点时间'''
    print(getlength(CurrentRouteDict))
    Pointsleft, NewRouteDict = InsertPoints(CurrentRouteDict, PointswitchDict)
    print(getlength(NewRouteDict), len(Pointsleft), getlength(NewRouteDict)+len(Pointsleft))
    #FirstOptimizedRoute = InterRouteOptimizer.Initiate(NewRouteDict)
    lastonelist, EndRoute = InsertLeftover(NewRouteDict, Pointsleft)
    print(getlength(EndRoute))
    #MapDict(FinalRoute, lastonelist, Newcenter)
    print(len(lastonelist))

    while len(lastonelist):
        FinalOptimizedRoute = InterRouteOptimizer.Initiate(EndRoute)
        OptimizedCurrentRouteDict = getcurrentinfo2(FinalOptimizedRoute)
        lastonelist, EndRoute = InsertLeftover(OptimizedCurrentRouteDict, lastonelist)
        print(len(lastonelist))
    MapDict(EndRoute, lastonelist, initialcenter, 0)
    todataframe(EndRoute).to_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Map/测试0.xlsx')

    centercompdlist.append(sum(calculatecompd(EndRoute)))

    for i in range(1, 100):
        print("iteration", i)
        InitialRouteDict = EndRoute
        print(getlength(InitialRouteDict))
        initialcenter = getroutecenter(InitialRouteDict)
        RemainingRouteDict, Pointsswitchlist, comp, compd = getpoints(initialcenter, InitialRouteDict)

        if len(Pointsswitchlist) == switchnum: switchcount += 1
        else:
            switchnum = len(Pointsswitchlist)
            switchcount = 0

        #RemainingRouteDict = InterRouteOptimizer.Initiate(RemainingRouteDict)
        compdlist.append(sum(compd))
        print(getlength(RemainingRouteDict), len(Pointsswitchlist), getlength(RemainingRouteDict) + len(Pointsswitchlist))
        #MapDict(RemainingRouteDict, Pointsswitchlist, initialcenter)
        #Newcenter = getroutecenter(RemainingRouteDict)
        #rand = np.random.random()
        #if rand >0.4: centerused = initialcenter
        #else: centerused = Newcenter
        #centerused = Newcenter
        centerused = initialcenter
        PointswitchDict = Sortdist(centerused, Pointsswitchlist)
        CurrentRouteDict = getcurrentinfo2(RemainingRouteDict)  ###'''返回的值中一行数据的倒数第二为到前点的距离，倒数第一为到前点时间'''
        print(getlength(CurrentRouteDict))
        Pointsleft, NewRouteDict = InsertPoints(CurrentRouteDict, PointswitchDict)
        print(getlength(NewRouteDict), len(Pointsleft), getlength(NewRouteDict) + len(Pointsleft))
        #FirstOptimizedRoute = InterRouteOptimizer.Initiate(NewRouteDict)
        #FirstOptimizedRoute = getcurrentinfo2(FirstOptimizedRoute)
        lastonelist, EndRoute = InsertLeftover(NewRouteDict, Pointsleft)
        print(getlength(EndRoute))

        print(len(lastonelist))
        while len(lastonelist):
            FinalOptimizedRoute = InterRouteOptimizer.Initiate(EndRoute)
            OptimizedCurrentRouteDict = getcurrentinfo2(FinalOptimizedRoute)
            lastonelist, EndRoute = InsertLeftover(OptimizedCurrentRouteDict, lastonelist)
            print(len(lastonelist))
        MapDict(EndRoute, lastonelist, centerused, i)
        todataframe(EndRoute).to_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Map/测试%s.xlsx' %i)
        if switchcount > 4:
            InitialRouteDict = EndRoute
            initialcenter = getroutecenter(InitialRouteDict)
            RemainingRouteDict, Pointsswitchlist, comp, compd = getpoints(initialcenter, InitialRouteDict)
            compdlist.append(sum(compd))
            break
        elif i == 99:
            InitialRouteDict = EndRoute
            initialcenter = getroutecenter(InitialRouteDict)
            RemainingRouteDict, Pointsswitchlist, comp, compd = getpoints(initialcenter, InitialRouteDict)
            compdlist.append(sum(compd))

        centercompdlist.append(sum(calculatecompd(EndRoute)))

    print(compdlist)
    print(compdlist.index(min(compdlist)))
    print(centercompdlist)




Initiate('morning', morning0)