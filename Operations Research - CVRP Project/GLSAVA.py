import itertools
import numpy as np
import pandas as pd
import non_overlap_optimizer
from geopy.distance import geodesic
import folium
#import Ruin_and_Restructure



distmatrix = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = 'Sheet1', index_col= 0 ))
pointdf = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集.xlsx',sheet_name = '点位表', header=None))
pointlist = pointdf.values.tolist()
df_Customers = pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx', sheet_name='表1点位表')
penaltydict = {}
globalcenterlist = []

def todict(data1):
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
        elif i[1] == '停车场' or i[1] == '转运站': continue
        else:
            list.append(i)
    return dict

def two_opt(a, i, j):
    if i == 0:
        return a[j:i:-1] + [a[i]] + a[j + 1:]
    return a[:i] + a[j:i - 1:-1] + a[j + 1:]


def cross(a, b, i, j):
    return a[:i] + b[j:], b[:j] + a[i:]


def insertion(a, b, i, j):
    #print(a, b, i, j)
    if len(a) == 0:
        return a, b
    while i >= len(a):
        i -= len(a)
    return a[:i] + a[i + 1:], b[:j] + [a[i]] + b[j:]


def swap(a, b, i, j):
    #print(a, b, i, j)
    if i >= len(a) or j >= len(b):
        return a, b
    a, b = a.copy(), b.copy()
    a[i], b[j] = b[j], a[i]
    return a, b


def fitnessfunc(route):
    totaldistance = 0
    for i in range(0, len(route)-1):
        current = route[i]
        next = route[i+1]
        distance = distmatrix.loc[current[0], next[0]]
        totaldistance += distance
    return totaldistance

def non_aug_obj_func(Routedict):
    routecost = 0
    overlapcost = 0
    penaltycost = 0
    for x in range(len(Routedict)):
        routecost += fitnessfunc(Routedict[x][1])
        for y in range(x + 1, len(Routedict)):
            overlapcost += non_overlap_optimizer.executepoly(Routedict[x][1], Routedict[y][1])
    objective = routecost * (1 + overlapcost)
    return objective

def obj_func(Routedict):
    print("objfunc")
    beta = 100
    row = 0.3
    routecost = 0
    overlapcost = 0
    penaltycost = 0
    for x in range(len(Routedict)):
        routecost += fitnessfunc(Routedict[x][1])
        for y in range(x + 1, len(Routedict)):
            overlapcost+= non_overlap_optimizer.executepoly(Routedict[x][1], Routedict[y][1])
        for point in Routedict[x][1]:
            pointcode = point[0]
            if penaltydict[x][pointcode]:
                penaltycost += penaltydict[x][pointcode] * geodesic((point[4], point[3]), (globalcenterlist[x][1], globalcenterlist[x][0])).km
            #加入penaltycost
    objective = routecost * (1+beta*overlapcost) + row*penaltycost
    return objective

def optimize(Routedict):
    print("optimize")
    Newdict = Routedict.copy()
    for i in range(len(Newdict)):
        is_stucked = False
        while not is_stucked:
            route = Newdict[i][1]
            is_stucked = True
            for k, j in itertools.combinations(range(len(route)), 2):
                new_route = two_opt(route, k, j)
                fitold = fitnessfunc(route)
                fitnew = fitnessfunc(new_route)
                if fitnew < fitold:
                    Newdict[i][1] = new_route
                    is_stucked = False
    return Newdict

def getroutecenter(route):
    longitude = 0
    latitude = 0
    i = 0
    for count in range(0, len(route) - 1):
        if route[count][1] == '停车场' or route[count][1] == '转运站':
            continue
        else:
            longitude += float(route[count][3])
            latitude += float(route[count][4])
            i += 1
    longitudebar = longitude / i
    latitudebar = latitude / i
    return(longitudebar, latitudebar)

def fitnessbetweenroute(leastroutecost, route1, route2, route1count, route2count):
    beta = 10
    row = 0.8
    lambda1 =1.5
    penaltycost = 0

    #传current best，当routecost大于best就直接结束

    routecost = fitnessfunc(route1)
    routecost += fitnessfunc(route2)
    #print("routecost:", routecost)
    if routecost > leastroutecost: return 10000000

    overlapcost = non_overlap_optimizer.executepoly(route1, route2)
    #print("overlap cost:", overlapcost)

    center1 = getroutecenter(route1)
    center2 = getroutecenter(route2)
    #把penalty cost单独拉出来单次计算
    for point in route1:
        pointcode = point[0]
        if penaltydict[route1count][pointcode]:
            penaltycost += (1+lambda1*penaltydict[route1count][pointcode]) * geodesic((point[4], point[3]),(center1[1], center1[0])).km
    for point in route2:
        pointcode = point[0]
        if penaltydict[route2count][pointcode]:
            penaltycost += (1+lambda1*penaltydict[route2count][pointcode]) * geodesic((point[4], point[3]),(center2[1], center2[0])).km
    #print("penalty cost:", penaltycost)
    fitness = routecost * (1 + beta * overlapcost) + row * penaltycost

    return fitness

def isfeasible(route, weightconst, timeconst):
    result = False
    weight = 0
    time = 0
    for i in route:
        weight += i[5]
        time += i[6]/60
    time += fitnessfunc(route)/(15000)
    #print(weight, weightconst)
    #print(time, timeconst)
    if weight<=weightconst and time <= timeconst:
        result = True
    #print("new weight", weight, "weight const", weightconst)
    #print("new time", time, "time const", timeconst)
    return result

def perturbation(Routedict):
    print("perturbation")
    best = Routedict
    is_stucked = False

    while not is_stucked:
        is_stucked = True
        for i, j in itertools.combinations(range(len(best)), 2): #在所有线路中获得两条线路进行优化
            print(i, j)
            #if Ruin_and_Restructure.executepoly(best[i][1], best[j][1]):
            carinfoi = best[i][0]
            weightconsti = carinfoi[4]
            carinfoj = best[j][0]
            weightconstj = carinfoj[4]
            timeconst = carinfoj[-1]
            bestfitness = fitnessbetweenroute(10000000,best[i][1], best[j][1], i, j)
            #bestfitness = fitnessfunc(best[i][1]) + fitnessfunc(best[j][1])
            for k, l in itertools.product(range(len(best[i][1])+1), range(len(best[j][1])+1)):
                for func in [cross, insertion, swap]:
                    c1, c2 = func(best[i][1], best[j][1], k, l)
                    if isfeasible(c1, weightconsti, timeconst) and isfeasible(c2, weightconstj, timeconst):
                        newfitness = fitnessbetweenroute(bestfitness, c1, c2, i, j)
                        #newfitness = fitnessfunc(c1) + fitnessfunc(c2)
                        #print(newfitness)
                        #print(bestfitness)
                        if  newfitness < bestfitness: #考虑计算视觉合理性
                            print(func)
                            print(k,l)
                            print("update")
                            best[i][1] = c1
                            best[j][1] = c2
                            bestfitness = newfitness
                            is_stucked = False
            else: continue
    return best

def getcenter(routedict):
    print("getcenter")
    center = []
    for routecount in range(0, len(routedict)):
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
        global globalcenterlist
        globalcenterlist = center
    return 0


def executeILS(Routedict):
    best = optimize(Routedict)
    print("Local search solution:")
    is_stucked = False
    objbest = obj_func(best)
    while not is_stucked:
        getcenter(Routedict)
        is_stucked = True
        new_solution = perturbation(best)
        new_solution = optimize(new_solution)
        objnew = obj_func(new_solution)
        print("new obj", objnew)
        if objnew < objbest:
            is_stucked = False
            best = new_solution
            objbest = objnew
        print("best dist", objbest)
    return best

def initiatepenalty(routedict):
    for routenum in range(len(routedict)):
        penalty = dict.fromkeys(pointlist[0], 0)
        penaltydict[routenum] = penalty
    return 0

def update_penalties(routedict):
    global penaltydict
    centerlist = globalcenterlist
    for routenum in range(0, len(routedict)):
        routeinfo = routedict[routenum][1]
        for point in routeinfo:
            #point = routeinfo[i] #当前点位
            if point[1] == '停车场' or point[1]  == '转运站':
                continue
            else:
                distancelist = []
                for j in centerlist:
                    distancelist.append(geodesic((point[4], point[3]), (j[1], j[0])).km)
                closetroute = distancelist.index(min(distancelist))
                if  closetroute != routenum:
                    penaltydict[routenum][point[0]] += 1
    return 0

def MapDict(Routedict, center, integer):
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
            if routeinfo[point][1]!='停车场':
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
    for point in center:
        folium.CircleMarker(location=[point[1], point[0]],
                            radius=3,
                            color='green',
                            tooltip=str(point[1])).add_to(m)

    m
    m.save('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Map/测试%s.html'%integer)
    return 0

def getcurrentinfo(route):
    weight = 0
    time = 0
    for i in route:
        weight += i[5]
        time += i[6]/60
    time += fitnessfunc(route)/(15000)
    return weight, time


def todataframe(routedict):
    data = pd.DataFrame(columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])
    for routecount in routedict:
        carinfo = routedict[routecount][0]
        routeinfo = []
        routes = routedict[routecount][1]
        for point in routes:
            routeinfo.append(point[:10])
        weight, time= getcurrentinfo(routes)
        routeinfo.append(['点位数量',len(routeinfo),carinfo[0], carinfo[1], weight, carinfo[3], carinfo[4], carinfo[5], time, carinfo[7], carinfo[8]])
        data = pd.concat([data, pd.DataFrame(routeinfo, columns = ['点位编号', '点位名称', '所属路街', '经度', '纬度', '垃圾量（升/处）', '作业停留时间（分钟）',
               '允许时间窗口', '禁止时间窗口', '收运频次（次/日）', '车辆通行条件'])], axis = 0)
    return data

def executeGLS(Routedict):
    initiatepenalty(Routedict)
    best = Routedict
    solutions = []
    solutionobj = []
    solutions.append(best)
    solutionobj.append(non_aug_obj_func(best))
    k = 0
    while k<20:
        local_min = executeILS(best).copy()
        update_penalties(local_min)
        solutions.append(local_min)
        solutionobj.append(non_aug_obj_func(local_min))
        todataframe(local_min).to_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Map/测试%s.xlsx' %k)
        MapDict(local_min, globalcenterlist, k)
        best = local_min
        k += 1
    print(solutionobj)
    return min(solutionobj)

'''
program开始
'''
#test = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件0（优化前）.xlsx', index_col=0))
#afternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/afternoon0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
#morning0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
#test1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Overlap重构优化结果/morning0/优化后New.xlsx', index_col=0))
#test2 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Overlap重构优化结果/afternoon0/优化前.xlsx', index_col=0))
test3 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/Overlap优化结果/afternoon0/优化后.xlsx', index_col=0))

testdict = todict(test3)
executeGLS(testdict)
