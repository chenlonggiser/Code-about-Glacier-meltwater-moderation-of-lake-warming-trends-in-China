import pandas as pd

# 导入数据
df = pd.read_excel('D:\\Datasets\\lake_temperature_test\\流域内气象因子平均值\\random_forest.xls', header=0)
# print(df.head())
# df = df[df['type']==1]
df = df.drop(columns=['type', 'rate'])

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
# spring_data = df[df['季节'] == '春季']

# # 筛选出夏季的数据
# summer_data = df[df['季节'] == '夏季']

# # 筛选出秋季的数据
# autumn_data = df[df['季节'] == '秋季']

# # 筛选出冬季的数据
# winter_data = df[df['季节'] == '冬季']

# df = df.groupby('Year').mean()
# df = df.iloc[:,1:]

# print(df)


# 检查是否有缺失值
# print(df.isnull().sum())

# 可视化数据集中不同特征之间的相关性。
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
# scatterplotmatrix(df.values, figsize=(100, 100),
#                   names=df.columns, alpha=0.5)
# plt.tight_layout()
# plt.show()

# 接下来，我们创建一个各变量之间的相关矩阵以量化线性关系。并绘制成热力图。
import numpy as np
from mlxtend.plotting import heatmap
# cm = np.corrcoef(df.values.T)
# hm = heatmap(cm, row_names=df.columns, column_names=df.columns, figsize=(10, 10))
# plt.tight_layout()
# plt.show()

# 首先我们将7个气象要素设置成预测（解释）变量，或者叫特征。LSWT设置为响应变量，或者叫目标变量。然后拆分数据集，前70%的数据作为训练集，后30%作为测试集。
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
from six import StringIO
from IPython.display import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import os

target = 'lswt'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

X = df.drop(columns=['lswt'])
y = df['lswt']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)
# 接下来，我们查看数据是否符合正态分布。
# 正偏态分布图
# sns.distplot(df['lswt'], color='green')
# plt.show()

# print("偏度为 %f " % df['lswt'].skew())
# print("峰度为 %f" % df['lswt'].kurt())

# 我们开始训练模型，并查看其平均绝对误差（MAE）与决定系数（R2）。
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=100,
                               criterion='squared_error',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')

# 如果测试集的结果明显不如训练集 我们再看看残差
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

# ax1.scatter(y_test_pred, y_test_pred - y_test,
#             c='limegreen', marker='s', edgecolor='white',
#             label='Test data')
# ax2.scatter(y_train_pred, y_train_pred - y_train,
#             c='steelblue', marker='o', edgecolor='white',
#             label='Training data')
# ax1.set_ylabel('Residuals')

# for ax in (ax1, ax2):
#     ax.set_xlabel('Predicted values')
#     ax.legend(loc='upper left')
#     ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)

# plt.tight_layout()

#plt.savefig('figures/09_16.png', dpi=300)
# plt.show()
score = forest.score(X_test, y_test)
# print('随机森林模型得分：', score)

# 查看预测值与真实值偏差
y_validation_pred = forest.predict(X_test)
# plt.figure()
# plt.plot(np.arange(128), y_test[:128], "go-", label="True value")
# plt.plot(np.arange(128), y_validation_pred[:128], "ro-", label="Predict value")
# plt.title("True value And Predict value")
# plt.legend()
# plt.show()

# 我们再从其它角度看看回归性能
# 评估回归性能
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_validation_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_validation_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_validation_pred)))

df_output = pd.DataFrame(columns=['AT','SP', 'SSRD', 'STRD', 'TE', 'TP', 'U_wind', 'V_wind', 'y_true', 'y_pred'])

# df_output['AT'] = X_test['AT']
df_output['SP'] = X_test['SP']
df_output['SSRD'] = X_test['SSRD']
df_output['STRD'] = X_test['STRD']
df_output['TE'] = X_test['TE']
df_output['TP'] = X_test['TP']
df_output['U_wind'] = X_test['U_wind']
df_output['V_wind'] = X_test['V_wind']

df_output['y_true'] = y_test
df_output['y_pred'] = y_validation_pred
df_output.to_excel('result_Y_validation.xlsx')

# 最后，我们看看各气象要素对LSWT的贡献率
pipe = Pipeline([('scaler', StandardScaler()), ('reduce_dim', PCA()),
                 ('regressor', forest)])
with open('./wine.dot','w',encoding='utf-8') as f:
    f=export_graphviz(pipe.named_steps['regressor'].estimators_[0], out_file=f)
    f=export_graphviz(pipe.named_steps['regressor'].estimators_[0], out_file=f)
col = list(X_train.columns.values)
importances = forest.feature_importances_
x_columns = ['AT','SP', 'SSRD', 'STRD', 'TE', 'TP', 'U_wind', 'V_wind']

indices = np.argsort(importances)[::-1]
list01 = []
list02 = []
for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, col[indices[f]], importances[indices[f]]))
    list01.append(col[indices[f]])
    list02.append(importances[indices[f]])
    
from pandas.core.frame import DataFrame

c = {"columns": list01, "importances": list02}
data_impts = DataFrame(c)
data_impts.to_excel('data_importances.xlsx')

importances = list(forest.feature_importances_)
feature_list = list(X_train.columns)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

import matplotlib.pyplot as plt

x_values = list(range(len(importances)))
print(x_values)
plt.figure(figsize=(10, 6))
plt.bar(x_values, importances, orientation='vertical', color=['#E08985','#D3D3D3', '#DEBF80', '#DCCD5B', '#73AD96', '#5F9069', '#6179A7', '#375631'], width=0.5)
plt.xticks(x_values, feature_list, rotation=96)
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.ylim([0, 0.6])
# plt.title('Variable Importances')
plt.show()
