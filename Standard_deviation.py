import pandas as pd
import numpy as np
import statistics
# 读取包含年份和湖泊温度数据文件
path = 'D:\\Datasets\\lake_temperature_test\\lt.csv'
df = pd.read_csv(path, header=0)

# def get_season(month):
#     if month in [3, 4, 5]:
#         return '春季'
#     elif month in [6, 7, 8]:
#         return '夏季'
#     elif month in [9, 10, 11]:
#         return '秋季'
#     else:
#         return '冬季'

# # 添加一列表示季节
# df['季节'] = df['Month'].apply(get_season)

# # 筛选出春季的数据
# spring_data = df[df['季节'] == '春季'].groupby('Year').mean().std()
# # print(spring_data)

# # 筛选出夏季的数据
# summer_data = df[df['季节'] == '夏季'].groupby('Year').mean().std()

# # 筛选出秋季的数据
# autumn_data = df[df['季节'] == '秋季'].groupby('Year').mean().std()

# # 筛选出冬季的数据
# winter_data = df[df['季节'] == '冬季'].groupby('Year').mean().std()

# save_path = 'D:\\Datasets\\lake_temperature_test\\noice_lake_temperature_seasonal_STd.xlsx'
# with pd.ExcelWriter(save_path) as writer:
#     spring_data.to_excel(writer, sheet_name='春季')
#     summer_data.to_excel(writer, sheet_name='夏季')
#     autumn_data.to_excel(writer, sheet_name='秋季') 
#     winter_data.to_excel(writer, sheet_name='冬季')



data = df.groupby('Year').mean().std()
# data = statistics.stdev(data)

save_path = 'D:\\Datasets\\lake_temperature_test\\LSWT_STD_year.csv'
data.to_csv(save_path)
print(data)