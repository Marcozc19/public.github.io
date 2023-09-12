import itertools
import pandas as pd

distmatrix = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx',sheet_name = 'Sheet1', index_col= 0 ))

df_Customers = pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx', sheet_name='表1点位表')
carlist = pd.DataFrame(pd.read_excel('C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新.xlsx', sheet_name='表4收运车辆表'))
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


def fitnessfunc(route):
    totaldistance = 0
    for i in range(0, len(route)-1):
        current = route[i]
        next = route[i+1]
        if current[1] == '停车场' or next[1] == '转运站':
            continue
        distance = distmatrix.loc[current[0], next[0]]
        totaldistance += distance
    return totaldistance

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
                    #print("update")
                    Newdict[i][1] = new_route
                    is_stucked = False
    return Newdict