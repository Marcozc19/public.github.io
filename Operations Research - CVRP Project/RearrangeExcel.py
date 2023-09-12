import random
from geopy.distance import geodesic
import openpyxl
import pandas as pd
from operator import itemgetter



cwpair_path = 'C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集.xlsx'

cwpairwb = openpyxl.load_workbook(cwpair_path)
alldist = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表2点位距离和时间表'))
facility = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表3设施表'))
facilitylist = facility.values.tolist()
car = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表4收运车辆表'))



alldistlist = alldist.values.tolist()
allcode = pd.DataFrame(pd.read_excel(cwpair_path, sheet_name='表1点位表'))

uniquecodedict = {}
uniquecodelist = []

name = allcode["序号"].values.tolist()
longitude = []
latitude = []

for code in name:
    if code in uniquecodelist: continue
    else:
        uniquecodelist.append(code)
        pointlong = allcode[allcode['序号'].isin([code])].values.tolist()[0][3]
        pointlat = allcode[allcode['序号'].isin([code])].values.tolist()[0][4]
        longitude.append(pointlong)
        latitude.append(pointlat)
        uniquecodedict[code] = [allcode[allcode['序号'].isin([code])].values.tolist()[0][3],allcode[allcode['序号'].isin([code])].values.tolist()[0][4]]
print(uniquecodedict)

Rest = allcode.iloc[:,5:]

random.shuffle(longitude)
random.shuffle(latitude)

count = 0
for point in uniquecodedict:
    uniquecodedict[point] = [longitude[count],latitude[count]]
    count+=1

print(uniquecodedict)
newlongitude = []
newlatitude = []

for point in name:
    coordinate = uniquecodedict[point]
    newlongitude.append(coordinate[0])
    newlatitude.append(coordinate[1])

Newallcode = pd.DataFrame({"序号": name, "名称": name, "街道":name, "经度": newlongitude, "纬度": newlatitude})
Newallcode = pd.concat([Newallcode, Rest],axis=1,sort=False)

nameseries = pd.Series(name)
uniquename = nameseries.unique()
Newdistlist = []
for i in uniquename:
    iinfo = Newallcode[Newallcode['名称'].isin([i])].values.tolist()[0]
    for j in uniquename:
        jinfo = Newallcode[Newallcode['名称'].isin([j])].values.tolist()[0]
        distinfo = [i, i, j, j, iinfo[3], iinfo[4], jinfo[3], jinfo[4],geodesic((iinfo[4], iinfo[3]), (jinfo[4], jinfo[3])).m, 356]
        Newdistlist.append(distinfo)
    for fac in facilitylist:
        distinfo = [i, i,fac[0], fac[1] , iinfo[3], iinfo[4], fac[4], fac[5], geodesic((iinfo[4], iinfo[3]), (fac[5],fac[4])).m, 356]
        Newdistlist.append(distinfo)
for fac in facilitylist:
    for j in uniquename:
        jinfo = Newallcode[Newallcode['名称'].isin([j])].values.tolist()[0]
        distinfo = [ fac[0],fac[1], j, j, fac[4], fac[5], jinfo[3], jinfo[4],geodesic((jinfo[4], jinfo[3]), (fac[5],fac[4])).m, 356]
        Newdistlist.append(distinfo)




Newdistdf = pd.DataFrame(Newdistlist, columns=['起点名称','起点编码','终点名称','终点编码','起点经度','起点纬度','终点经度','终点纬度','drivedistance','时间(秒)'])
#print(Newdistdf)



allcodelist = allcode.values.tolist()
allcodelist = sorted(allcode.values.tolist(), key=itemgetter(0))

writer = pd.ExcelWriter("C:/Users/zhuan/Desktop/清华大学/2022毕业论文/数据集/环卫车数据集新新.xlsx")
Newallcode.to_excel(writer, sheet_name='表1点位表')
Newdistdf.to_excel(writer, sheet_name='表2点位距离和时间表')
facility.to_excel(writer, sheet_name='表3设施表')
car.to_excel(writer, sheet_name='表4收运车辆表')
writer.save()
writer.close()