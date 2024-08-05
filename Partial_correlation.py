import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy import stats

data_path= 'D:\\Datasets\\lake_temperature_test\\ERA5\\noice_lake\\random_forest_data.xlsx'
data = pd.read_excel(data_path, header=0)
data = data.groupby('Year').mean()
# x = data['lswt']
# y = data['AT']
# z = data['STRD']

# 选择要分析的变量
variables_of_interest = ['lswt', 'AT', 'STRD']  # 假设 'X' 和 'Y' 是目标变量，'Z' 是控制变量

# 添加常数列用于计算VIF
data_with_const = add_constant(data[variables_of_interest])

# 计算偏相关系数
def calculate_partial_corr(data, x, y, control_vars):
    # 回归 x 对控制变量
    x_model = sm.OLS(data[x], data[control_vars]).fit()
    # 回归 y 对控制变量
    y_model = sm.OLS(data[y], data[control_vars]).fit()
    # 计算回归残差
    x_resid = x_model.resid
    y_resid = y_model.resid
    # 计算残差之间的相关性
    partial_corr = np.corrcoef(x_resid, y_resid)[0, 1]
    return partial_corr

# 计算 X 和 Y 之间的偏相关系数，控制 Z 的影响
x = 'lswt'
y = 'AT'
control_vars = ['const', 'STRD']  # 添加 'const' 用于常数项
partial_corr = calculate_partial_corr(data_with_const, x, y, control_vars)

print(f"控制变量 {control_vars[1:]} 后，{x} 和 {y} 之间的偏相关系数为: {partial_corr}")

# 可视化偏相关分析结果（可选）
import seaborn as sns
import matplotlib.pyplot as plt

# 计算 t 统计量
n = len(data)
k = len(control_vars) - 1
t_stat = partial_corr * np.sqrt((n - k - 2) / (1 - partial_corr**2))
# 计算 p 值
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k - 2))
t_stat = partial_corr * np.sqrt((n - k - 2) / (1 - partial_corr**2))
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k - 2))

print(f"t 统计量: {t_stat}")
print(f"p 值: {p_value}")



# 可视化 X 和 Y 的关系，控制 Z 的影响
norm = Normalize(vmin=data['AT'].min(), vmax=data['AT'].max())
cmap = cm.viridis
plt.figure(figsize=(10, 6))
# 添加回归线
sns.regplot(x='lswt', y='AT', data=data, scatter=False, color='black')

sc = plt.scatter(data['lswt'], data['AT'], c=data['STRD'], cmap=cmap, norm=norm)
# plt.title(f'{x} vs {y} (controlled for {control_vars[1:]})')
plt.xlabel('LSWT (℃)')
plt.ylabel('AT (℃)')
plt.colorbar(sc, label='STRD (W/m²)')
plt.show()
