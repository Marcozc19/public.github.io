import pandas as pd
from geopy.distance import geodesic
import non_overlap_optimizer

distmatrix = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = 'Sheet1', index_col= 0 ))
global geocenter
geocenter = []
global routecenter
routecenter = []


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

def fitnessfunc(route):
    totaldistance = 0
    for i in range(0, len(route)-1):
        current = route[i]
        next = route[i+1]
        if current[1] == '停车场' or next[1]=='转运站': continue
        distance = distmatrix.loc[current[0], next[0]]
        totaldistance += distance
    return totaldistance

def non_aug_obj_func(Routedict):
    routecost = 0
    overlapcost = 0
    beta = 5
    for x in range(len(Routedict)):
        routecost += fitnessfunc(Routedict[x][1])
        for y in range(x + 1, len(Routedict)):
            overlapcost += non_overlap_optimizer.executepoly(Routedict[x][1], Routedict[y][1])
            # 加入penaltycost
    objective = routecost * (1 + beta*overlapcost)
    return objective

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
    global geocenter
    geocenter = center
    return 0


def getroutecenter(routedict):
    center = []
    for routecount in range(0, len(routedict)):
        routeinfo = routedict[routecount][1]
        length = len(routeinfo)
        mid = int(length/2)
        midpoint =  routeinfo[mid]
        center.append([midpoint[3], midpoint[4]])
        global routecenter
        routecenter = center
    return 0

def main(file):
    solutiondict = todict(file)
    print("overall objective:", non_aug_obj_func(solutiondict))
    totaldist = 0
    for routecount in range(len(solutiondict)):
        route = solutiondict[routecount][1]
        totaldist += fitnessfunc(route)
    print("total distance:", totaldist)

    getroutecenter(solutiondict)
    getcenter(solutiondict)
    compd = []
    compc = []
    for routecount in range(len(solutiondict)):
        route = solutiondict[routecount][1]
        for point in route:
            totaldistc = 0
            totaldistd = 0
            if point[1] != '停车场' and point[1]  != '转运站':
                totaldistc += ((geodesic((point[4], point[3]), (geocenter[routecount][1], geocenter[routecount][0])).km))
                totaldistd += ((geodesic((point[4], point[3]), (routecenter[routecount][1], routecenter[routecount][0])).km))
            compc.append(totaldistc/len(route))
            compd.append(totaldistd/len(route))



    print("compc:", sum(compc))
    print("compd:", sum(compd))

    sumall = []
    for i in range(len(solutiondict)):
        for j in range(i+1, len(solutiondict)):
            area = non_overlap_optimizer.getarea(solutiondict[i][1], solutiondict[j][1])
            sumall.append(area)
    print("overlap area:", float(sum(sumall)*100000))
    return 0
'''
afternoon0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_下午且条件0（优化前）.xlsx', index_col=0))
morning0 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/output_test_上午且条件0（优化前）.xlsx',index_col=0))
#GLStest = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/GLS优化后/使用aug_obj/测试0.xlsx', index_col=0))
test1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/morning0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
test2 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/紧凑性优化结果/afternoon0/型凑性优化——成本优化后/优化前.xlsx', index_col=0))
test5 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Overlap重构优化结果/morning0/优化后.xlsx', index_col=0))
test3 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/Overlap重构优化结果/afternoon0/优化前.xlsx', index_col=0))

test4 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/GLSAV结果/Afternoon0/测试0.xlsx', index_col=0))
test6 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/GLSAV结果/Morning0/测试1.xlsx', index_col=0))
test7 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/GLS结果/morning0/测试17.xlsx', index_col=0))
test8 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/GLS结果/afternoon0/测试5.xlsx', index_col=0))
'''
test = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/GLS/测试0.xlsx', index_col=0))
test1 = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/输出结果/案例二/GLSAVA/测试0.xlsx', index_col=0))
main(test)
main(test1)



