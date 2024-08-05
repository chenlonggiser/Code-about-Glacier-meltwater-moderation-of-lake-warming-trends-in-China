import pandas as pd
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import theilslopes
# 读取Excel数据
path = 'D:\\nature_water\\catchment_glaciated_lakes_temperature.xlsx'
df = pd.read_excel(path, header=0, index_col=[0])
# df = df.mean(axis=1)
# print(df)
# save_path = 'D:\\Datasets\\lake_temperature_test\\noice_lake_temperature_mean.csv'
# df.to_csv(save_path)
# 根据年份和月份计算季节
def get_season(month):
    if month in [3, 4, 5]:
        return '春季'
    elif month in [6, 7, 8]:
        return '夏季'
    elif month in [9, 10, 11]:
        return '秋季'
    else:
        return '冬季'

# 添加一列表示季节
df['季节'] = df['Month'].apply(get_season)

# 筛选出春季的数据
spring_data = df[df['季节'] == '春季'].groupby('Year').mean()
spring_data = spring_data.mean(axis=1)
# print(spring_data)

# 筛选出夏季的数据
summer_data = df[df['季节'] == '夏季'].groupby('Year').mean()
summer_data = summer_data.mean(axis=1)

# 筛选出秋季的数据
autumn_data = df[df['季节'] == '秋季'].groupby('Year').mean()
autumn_data = autumn_data.mean(axis=1)

# 筛选出冬季的数据
winter_data = df[df['季节'] == '冬季'].groupby('Year').mean()
winter_data = winter_data.mean(axis=1)

# 求出年平均值
Year_data = df.drop('Month', axis=1)
Year_data = Year_data.groupby('Year').mean()

# 提取年份和温度数据
def Theil(data):
    years = data.index.values.reshape(-1, 1)  # 将年份数据转换为二维数组
    temperatures = data.values

    # 创建并训练TheilSen回归模型
    model = TheilSenRegressor()
    model.fit(years, temperatures)

    # 输出回归方程参数
    rate = model.coef_[0]
    intercept = model.intercept_
    return rate, intercept

spring = Theil(spring_data)
print(f'春季的升温速率:{spring[0]}')
print(f'春季的截距:{spring[1]}')

summer = Theil(summer_data)
print(f'夏季的升温速率:{summer[0]}')
print(f'夏季的截距:{summer[1]}')

autumn = Theil(autumn_data)
print(f'秋季的升温速率:{autumn[0]}')
print(f'秋季的截距:{autumn[1]}')

winter = Theil(winter_data)
print(f'冬季的升温速率:{winter[0]}')
print(f'冬季的截距:{winter[1]}')

# save_path = 'D:\\Datasets\\lake_temperature_test\\无冰川区域气温季节温度.xlsx'
# with pd.ExcelWriter(save_path) as writer:
#     spring_data.to_excel(writer, sheet_name='春季')
#     summer_data.to_excel(writer, sheet_name='夏季')
#     autumn_data.to_excel(writer, sheet_name='秋季') 
#     winter_data.to_excel(writer, sheet_name='冬季')
