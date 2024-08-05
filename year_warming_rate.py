import pandas as pd
import numpy as np
from sklearn.linear_model import TheilSenRegressor
# 读取包含年份和湖泊温度数据的CSV文件
path = 'D:\\nature_water\\LSWT_monthly.csv'
data = pd.read_csv(path,index_col=0)
# 计算每个湖泊的每一年的年平均温度
df = data.drop(['Month'], axis=1)
df = df.groupby('Year').mean()
# print(df.head())
# 计算每个湖泊的增温速率
years = df.index.values.reshape(-1, 1)  # 将年份数据转换为二维数组
cols = df.columns.to_list()
# print(years)
rate = []
# print(type(cols[0]), type(df.columns))
# print(df.columns)
# print(df[2])
for col in cols:
    temperatures = df[col]
    # 创建并训练TheilSen回归模型
    model = TheilSenRegressor()
    model.fit(years, temperatures)
    # print(model.coef_[0])
    rate.append(model.coef_[0]*10)#将升温速率换成10年的
rate = pd.Series(rate, index=cols)   
savepath = 'D:\\nature_water\\LSWT_rate.csv'
rate.to_csv(savepath)
print(rate)