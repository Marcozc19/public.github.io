import openpyxl
import pandas as pd
from geopy.distance import geodesic
import itertools
import sympy
from sympy import Point2D, Polygon
from scipy.spatial import ConvexHull
import folium

cwpair_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx'
cwpairwb = openpyxl.load_workbook(cwpair_path)

distmatrix = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = 'Sheet1', index_col= 0 ))
distancealllist = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = '表2点位距离和时间表'))
distinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表2点位距离和时间表'))
carinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表4收运车辆表'))
carinfolist = carinfo.values.tolist()
facilityinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表3设施表'))
demandinfo = pd.DataFrame(pd.read_excel(cwpair_path,sheet_name = '表1点位表'))



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

seedlist=[]

for demand in demandinfolist:
    if demand[7] == "5:00-9:00":
        if demand[-1] == 0:
            morning0.append(demand)
        else: morning1.append(demand)
    else:
        if demand[-1] == 0:
            afternoon0.append(demand)
        else: afternoon1.append(demand)

def MapDict(Routedict, center, integer):
    m = folium.Map(location=demandinfo[['纬度', '经度']].mean(),
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
        folium.CircleMarker(location=[point[4], point[3]],
                            radius=3,
                            color='green',
                            tooltip=str(point[1])).add_to(m)

    m
    m.save('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Map/Overlap测试%s.html'%integer)
    return 0


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

def insertleftover(Routedict, dummyfindpointlist):
    for points in dummyfindpointlist:
        BestInsertRoute = 100
        BestInsertIndex = 100
        Bestnewtime = 0
        mintimeincrease = 10000
        for routecount in range(0, len(Routedict)):  # 寻找每个点到每个线路中心的距离
            carinfo = Routedict[routecount][0]
            routeinfo = Routedict[routecount][1]
            weightconst = carinfo[6]
            timeconst = carinfo[-1]
            totalweight = carinfo[4]
            totaltime = carinfo[-3]
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
                totalweight = carinfo[4]
                insertinfo = points
                insertinfo.append(timefromprev*speed)
                insertinfo.append(timefromprev)
                routeinfo.insert(BestInsertIndex+1, insertinfo)
                carinfo[4] = totalweight + points[5] #更新线路需求
                carinfo[-3] = Bestnewtime #更新线路时间
                Routedict[BestInsertRoute][1] = routeinfo
                Routedict[BestInsertRoute][0] = carinfo
                break
    return Routedict




def getinitialroute(pointslist, carlist, seed1, seed2):
    dummypointlist = pointslist.copy() ##dummy点位list，会将被插入线路的点位删除
    centerlist = []
    routedict = {}
    routecount = 0
    pointcount = 0
    print(len(pointslist))
    dummypointlist.remove(seed1)
    dummypointlist.remove(seed2)
    #print("seed1:", seed1)
    #print("seed2:", seed2)

    routeinfo1 = [seed1]
    routeinfo2 = [seed2]
    car1 = carlist[0]
    carplate1 = car1[2]
    timeconst = car1[-1]
    weightconst1 = car1[6]
    totalweight1 = 0
    totaltime1 = 0
    car2 = carlist[1]
    carplate2 = car2[2]
    weightconst2 = car2[6]
    totalweight2 = 0
    totaltime2 = 0
    center1 = [seed1[3],seed1[4]]
    center2 = [seed2[3], seed2[4]]

    while(len(dummypointlist)):
        index1 = findmin(dummypointlist, center1)
        point1 = dummypointlist[index1]
        pointcode1 = point1[0]
        pointweight1 = point1[5]
        pointtime1 = point1[6]
        if totalweight1 + pointweight1 < weightconst1:
            insertindex, timefromprev, timetoafter, originaltime, speed = findinsertinterval(pointcode1, routeinfo1)
            newtime = totaltime1 + timefromprev + timetoafter + pointtime1 / 60 - originaltime
            if (newtime < timeconst):
                routeinfo1.insert(insertindex + 1, point1)
                totalweight1 += pointweight1
                totaltime1 = newtime
                #print("insert 1:" , point1)
                dummypointlist.pop(index1)
                pointcount += 1
        if not len(dummypointlist): break
        index2 = findmin(dummypointlist, center2)
        point2 = dummypointlist[index2]
        pointcode2 = point2[0]
        pointweight2 = point2[5]
        pointtime2 = point2[6]
        if totalweight2 + pointweight2 < weightconst2:
            insertindex, timefromprev, timetoafter, originaltime, speed = findinsertinterval(pointcode2, routeinfo2)
            newtime = totaltime2 + timefromprev + timetoafter + pointtime2 / 60 - originaltime
            if (newtime < timeconst):
                routeinfo2.insert(insertindex + 1, point2)
                #print("insert 2:",point2)
                totalweight2 += pointweight2
                totaltime2 = newtime
                dummypointlist.pop(index2)
                pointcount += 1
        print(len(dummypointlist))
        # print(totalweight + pointweight, weightconst)




    '''
    for i in range(len(carlist)):
        cars = carlist[i]
        dummyfindpointlist = dummypointlist.copy() ##dummydummylist, 在查找最近点位时，会删除那些已被尝试插入但未被插入的点位
        carplate = cars[0]
        timeconst = cars[-1]
        weightconst = cars[6]
        routeinfo = [seed]
        center = [seed[3],seed[4]]
        totalcenter = center.copy()
        totalweight = 0
        totaltime = 0

        while len(dummyfindpointlist):
            pointcount+=1
            index = findmin(dummyfindpointlist, center)
            point = dummyfindpointlist[index]
            dummyfindpointlist.pop(index)
            pointcode = point[0]
            pointcorrdinate = [point[3], point[4]]
            pointweight = point[5]
            pointtime = point[6]
            #print(totalweight + pointweight, weightconst)
            if totalweight + pointweight < weightconst:
                insertindex, timefromprev, timetoafter, originaltime, speed = findinsertinterval(pointcode, routeinfo)
                newtime = totaltime + timefromprev + timetoafter + pointtime/60 - originaltime
                if (newtime < timeconst):
                    routeinfo.insert(insertindex+1, point)
                    totalweight += pointweight
                    totaltime = newtime
                    dummypointlist.pop(index)
                    pointcount += 1
                    totalcenter[0] += pointcorrdinate[0]
                    totalcenter[1] += pointcorrdinate[1]
                    center[0] = totalcenter[0] / (len(routeinfo) - 1)
                    center[1] = totalcenter[1] / (len(routeinfo) - 1)
                else:
                    centerlist.append(center)
                    break
            else:
                centerlist.append(center)
                break
        #print(['点位数量', len(routeinfo), carplate, '现在载重', totalweight, '载重限制', weightconst, '现在时间', totaltime,'时间限制', timeconst])
        '''
    routedict[0] = [['点位数量', len(routeinfo1), carplate1, '现在载重', totalweight1, '载重限制', weightconst1, '现在时间', totaltime1,'时间限制', timeconst], routeinfo1]
    routedict[1] = [['点位数量', len(routeinfo2), carplate2, '现在载重', totalweight2, '载重限制', weightconst2, '现在时间', totaltime2,'时间限制', timeconst], routeinfo2]
    print(len(routedict[0][1]) + len(routedict[1][1]))
    #if len(dummypointlist):
     #   routedict = insertleftover(routedict, dummyfindpointlist)
    print(len(routedict[0][1])+len(routedict[1][1]))
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

def mergeroute(route1, route2):
    routeinfo1 = route1[1]
    routeinfo2 = route2[1]
    newset = routeinfo1.copy()
    newset.extend(routeinfo2)
    return newset

def findfurtherestpoint(set):
    furthestdist = 0
    for i in range(len(set)):
        for j in range(i, len(set)):
            dist = geodesic((set[i][4], set[i][3]), (set[j][4], set[j][3]))
            if dist > furthestdist:
                pointsetindex = [i,j]
                furthestdist = dist
    global seedlist
    seedlist = [set[pointsetindex[0]], set[pointsetindex[1]]]
    return set[pointsetindex[0]], set[pointsetindex[1]]

def executeoverlap(route1, route2):
    carinfolist = []
    carinfolist.append(route1[0])
    carinfolist.append(route2[0])
    pointlist = mergeroute(route1, route2)
    seed1, seed2 = findfurtherestpoint(pointlist)
    Routedict = getinitialroute(pointlist, carinfolist, seed1, seed2).copy()
    return Routedict[0],Routedict[1]


def polygon_intersection(p1: sympy.Polygon, p2: sympy.Polygon) -> sympy.Polygon:
    intersection_result = []

    assert isinstance(p1, sympy.Polygon)
    assert isinstance(p2, sympy.Polygon)

    k = p2.sides
    for side in p1.sides:
        if p2.encloses_point(side.p1):
            intersection_result.append(side.p1)
        if p2.encloses_point(side.p2):
            intersection_result.append(side.p2)
        for side1 in k:
            if p1.encloses_point(side1.p1):
                intersection_result.append(side1.p1)
            if p1.encloses_point(side1.p2):
                intersection_result.append(side1.p2)
            for res in side.intersection(side1):
                if isinstance(res, sympy.Segment):
                    intersection_result.extend(res.points)
                else:
                    intersection_result.append(res)

    intersection_result = list(sympy.utilities.iterables.uniq(intersection_result))

    if intersection_result:
        return sympy.Polygon(*intersection_result)
    else:
        return None

def calculateoverlapp(routemain, route1):
    hullmain = ConvexHull(routemain)
    hull1 = ConvexHull(route1)
    verticemainindex = hullmain.vertices.tolist()
    vertice1index = hull1.vertices.tolist()
    verticemain = []
    vertice1 = []
    for i in verticemainindex:
        verticemain.append(Point2D(routemain[i]))
    for j in vertice1index:
        vertice1.append((Point2D(route1[j])))
    verticemaintuple = tuple(verticemain)
    vertice1tuple = tuple(vertice1)
    #print(verticemain)
    #print(vertice1)
    polymain = Polygon(*verticemaintuple)
    poly1 = Polygon(*vertice1tuple)
    overlapploygon = polygon_intersection(polymain, poly1)
    return overlapploygon, polymain, poly1

def executepoly(route1, route2):
    route1location = []
    route2location= []
    for point in route1:
        if point[1] != '停车场' and point[1] != '转运站':
            route1location.append([point[3], point[4]])
    for point in route2:
        if point[1] != '停车场' and point[1] != '转运站':
            route2location.append([point[3], point[4]])

    overlap, polymain, poly1 = calculateoverlapp(route1location, route2location)
    if overlap:
        return 1
    else: return 0

def perturbation(Routedict):
    print("perturbation")
    best = Routedict.copy()
    is_stucked = False
    count = 0

    while not is_stucked:
        is_stucked = True
        for i, j in itertools.combinations(range(len(best)), 2): #在所有线路中获得两条线路进行优化
            print(i, j)
            if  executepoly(best[i][1], best[j][1]):
                newroute1, newroute2 = executeoverlap(best[i], best[j])
                best[i] = newroute1.copy()
                best[j] = newroute2.copy()
                MapDict(best, seedlist, count)
                count += 1
                if executepoly(best[i][1], best[j][1]):
                    print("still overlap")
                else: print("no overlap")
            else: continue
    return best

def todict(data1):
    datalist = data1.values.tolist()
    dict = {}
    count = 0
    list = []
    for i in datalist:
        if i[0]=='点位数量':
            dict[count] = []
            dict[count].append(i)
            dict[count].append(list)
            count += 1
            list = []
        elif i[1] == '停车场' or i[1] == '转运站': continue
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

def getcurrentinfo(route):
    weight = 0
    time = 0
    for i in route:
        weight += i[5]
        time += i[6]/60
    time += fitnessfunc(route)/(15000)
    return weight, time

'''
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
'''

#test = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化后.xlsx', index_col=0))
#afternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/afternoon0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
#morning0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
#test1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Overlap重构优化结果/morning0/优化后.xlsx', index_col=0))
test2 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/紧凑性优化结果/morning0/优化前.xlsx', index_col=0))

testdict = todict(test2)
#perturbation(testdict)
todataframe(perturbation(testdict)).to_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/紧凑性优化结果/morning0/优化后.xlsx')

