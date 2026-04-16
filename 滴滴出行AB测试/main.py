import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from scipy.stats import ttest_ind, mannwhitneyu, shapiro



# 1.数据加载

DATA_PATH = '/home/mw/input/didi4010/test.xlsx'

data = pd.read_excel(DATA_PATH)


# 2. 数据探索
# 查看实验组别
print(f"实验组别: {data['group'].unique()}")

# 对照组数据范围
control_data_range = data[data['group'] == 'control']['data'].agg(['min', 'max'])
print(f"对照组数据范围: {control_data_range}")

# 实验组时间范围
exp_date_range = data[data['group'] == 'experiment']['date'].agg(['min', 'max'])
print(f"实验组时间范围: {exp_date_range}")


# 3. 特征工程
# ROI = (gmv - 补贴成本) / 补贴成本
data['roi'] = (data['gmv'] - (data['coupon per trip'] * data['trips'])) / (data['coupon per trip'] * data['trips'])

# 转化率 = 订单数 / 用户数
data['conversion_rate'] = data['trips'] / data['users']

# 订单取消率 = 取消订单数 / 请求数
data['order_cancel_rate'] = data['canceled orders'] / data['requests']

# 4. 数据分组
control = data[data['group'] == 'control']
experiment = data[data['group'] == 'experiment']

print(f"控制组样本数: {len(control)}, 实验组样本数: {len(experiment)}")

# 5. 正态性检验与方差齐性检验
METRICS = ['requests', 'gmv', 'coupon per trip', 'trips', 'canceled requests', 'roi', 'conversion_rate', 'order_cancel_rate']

print("\n" + "=" * 60)
print("正态性检验 (Shapiro-Wilk) 与 方差齐性检验 (Levene)")
print("=" * 60)

for metric in METRICS:
    _, p_control = shapiro(control[metric])
    _, p_experiment = shapiro(experiment[metric])
    _, p_levene = st.levene(control[metric], experiment[metric])

    print(f"\n指标: {metric}")
    print(f"  正态性检验: 控制组 p={p_control:.4f}, 实验组 p={p_experiment:.4f}")
    print(f"  方差齐性检验: p={p_levene:.4f}")

    if p_control > 0.05 and p_experiment > 0.05 and p_levene > 0.05:
        print("  结论: 使用 t 检验")
    else:
        print("  结论: 使用非参数检验 (Mann-Whitney U)")

# 6. 假设检验 (t检验)
METRIC_PAIRS = [
    ('requests', 'requests'),
    ('gmv', 'gmv'),
    ('trips', 'trips'),
    ('canceled requests', 'canceled requests'),
    ('coupon per trip', 'coupon per trip'),
    ('roi', 'roi'),
    ('conversion_rate', 'conversion_rate'),
]

print("\n" + "=" * 60)
print("均值差异检验 (独立样本 t 检验)")
print("=" * 60)

for control_col, exp_col in METRIC_PAIRS:
    t_stat, p_value = ttest_ind(control[control_col], experiment[exp_col], equal_var=True)
    significance = "显著" if p_value < 0.05 else "不显著"
    print(f"{control_col}: p值={p_value:.4f} ({significance})")

# 7. 可视化分析
plt.figure(figsize=(10, 6))
plt.hist([control['order_cancel_rate'], experiment['order_cancel_rate']], bins=20, alpha=0.5)
plt.xlabel('Order Cancel Rate')
plt.ylabel('Frequency')
plt.legend(['Control', 'Experiment'])
plt.title('Distribution of Order Cancel Rate: Control vs Experiment')
plt.tight_layout()
plt.show()