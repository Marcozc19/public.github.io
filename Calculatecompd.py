import openpyxl
import csv
import numpy as np
import pandas as pd
from operator import itemgetter
from geopy.distance import geodesic
import random
import folium
import InterRouteOptimizer
import os

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


def todictnew(data1):
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