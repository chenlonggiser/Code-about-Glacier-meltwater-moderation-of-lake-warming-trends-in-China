import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy import stats

#第一个散点图
path = 'D:\\Datasets\\lake_temperature_test\\all_rate.csv'
df = pd.read_csv(path, header=0)

# x = df['Elevation(m)']
# y = df['rate']

x = df[df['type'] == 1]['Elevation(m)'].values
y = df[df['type'] == 1]['rate'].values

# Calculate the point density
xy = np.vstack([x,y])  #  将两个维度的数据叠加
z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(6, 6))
sc = plt.scatter(x, y,c=z, s=20,cmap='Spectral_r') # c表示标记的颜色

# 计算Pearson相关系数并绘制回归线
correlation = df.corr(method='pearson')
sns.regplot(x='Elevation(m)', y='rate', data=df, scatter=False, color='black')
# 显示相关系数和趋势检验
plt.text(0.8, 0.8, f'R = {correlation.loc["Elevation(m)", "rate"]:.2f}', horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Elevation(m)'], df['rate'])
plt.text(0.8, 0.9, f'P < 0.001', horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
plt.ylabel('Trend in LSWT (°C decade⁻¹)')
# plt.subplot(222)
# plt.subplot(224)
# 添加渐变图例
cbar = plt.colorbar(sc)
plt.show()

