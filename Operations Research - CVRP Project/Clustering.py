import ReadExcel
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

colo = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# x = np.random.randint(0, 10, [100, 2])
x = ReadExcel.Location.iloc[:, [5,6]]
shape = x.shape
K = 4   # 簇个数
print(x)
# 分析层次聚类
row_clusters = linkage(pdist(x, metric='euclidean'), method='complete')
'''
pdist：计算样本距离,其中参数metric代表样本距离计算方法
(euclidean:欧式距离、minkowski:明氏距离、chebyshev:切比雪夫距离、canberra:堪培拉距离)
linkage：聚类,其中参数method代表簇间相似度计算方法
(single:  MIN、ward：沃德方差最小化、average：UPGMA、complete：MAX)
'''
'''
row_dendr = dendrogram(row_clusters)
plt.tight_layout()
plt.title('canberra-complete')
plt.show()
'''
# 层次聚类
clf = AgglomerativeClustering(n_clusters=K, affinity='euclidean', linkage='complete')
labels = clf.fit_predict(x)
count = [0,0,0,0,0,0]
for i in labels:
    count[i]+=1
#print('cluster labels:%s' % labels)
print(count)
for k, col in zip(range(0, K), colo):
    X = labels == k
    plt.plot(x[X, 0], x[X, -1], col + 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('hierarchical_clustering')
plt.show()