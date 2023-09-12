import numpy as np
import sympy
from sympy import Point2D, Polygon
from scipy.spatial import ConvexHull
import pandas as pd

#import CompactnessOptimizer

#test = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化后.xlsx', index_col=0))
#afternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件0（优化前）.xlsx', index_col=0))
#afternoon1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件1（优化前）.xlsx',index_col=0))
#morning0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件0（优化前）.xlsx',index_col=0))
#morning1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件1（优化前）.xlsx',index_col=0))

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
        return float(abs(overlap.area)/(polymain.area+poly1.area))
    else: return 0

def todict(data1):
    datalist = data1.values.tolist()
    dict = {}
    count = 0
    list = []
    for i in datalist:
        if i[0]=='点位数量':
            carplate = i[2]
            weightconst = i[6]
            timeconst = i[-1]
            carinfo = [carplate, '现有载重',i[4], weightconst, '现在耗时', i[-3], timeconst]
            dict[count] = []
            dict[count].append(carinfo)
            dict[count].append(list)
            count += 1
            list = []
        else:
            list.append(i)
    return dict

def getarea(route1, route2):
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
        return float(overlap.area)
    else: return 0

def getoveralloverlap(file):
    InitialSolution = todict(file)
    sumall = []
    for i in range(len(InitialSolution)):
        for j in range(i+1, len(InitialSolution)):
            area = getarea(InitialSolution[i][1], InitialSolution[j][1])
            sumall.append(area)

    return float(sum(sumall)*100000)

#print(getoveralloverlap(afternoon0))



#for x in range(0, len(InitialSolution)):
#    for y in range(x+1, len(InitialSolution)):
#        print(x,"&", y, executepoly(Locationdict[x], Locationdict[y]))
