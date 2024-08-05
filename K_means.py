#利用k-means聚类算法实现鸢尾花数据（第3个和第4个维度）的聚类。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

path = 'D:\\Datasets\\lake_temperature_test\\all_rate.csv'
df = pd.read_csv(path)

X = df['Longitude', 'Latitude'] #只取后两个维度
#绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c = "red", marker='o', label='0')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='1')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='2')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='3')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()