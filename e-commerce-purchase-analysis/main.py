import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


# 1. 前期准备与配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 2. 数据加载
print("=" * 70)
print("电商购买行为分析及建模报告")
print("=" * 70)
# 读取CSV文件
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customerData_500k.csv')
df = pd.read_csv(csv_path)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

# 3. 查看数据质量
print("\n" + "=" * 70)
print("一、数据质量报告")
print("=" * 70)

missing_info = pd.DataFrame({
    '缺失数量': df.isnull().sum(),
    '缺失比例(%)': (df.isnull().sum() / len(df) * 100).round(2)
})

missing_info = missing_info[missing_info['缺失数量'] > 0]
if len(missing_info) > 0:
    print("\n缺失值情况:")
    print(missing_info)
else:
    print("\n✓ 数据无缺失值")

duplicates = df.duplicated().sum()
print(f"\n重复记录: {duplicates} 条 ({(duplicates/len(df)*100):.2f}%)")

# 对上一步的异常值检测

print("\n异常值检测 (IQR方法):")

outlier_summary = []
for col in ['Age', 'AnnualIncome']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    outlier_summary.append({'特征': col, '异常值数量': outliers, '异常比例(%)': f"{(outliers/len(df)*100):.2f}"})
    print(f"  {col}: {outliers} 个 ({(outliers/len(df)*100):.2f}%)")

# 4. 描述性统计分析，先查看数值型特征的描述性统计信息，大概了解数据的分布情况

print("\n" + "=" * 70)
print("二、描述性统计分析")
print("=" * 70)

print("\n数值型特征统计:")
num_stats = df[['Age', 'AnnualIncome', 'PurchaseStatus', 'CustomerSatisfaction']].describe().T
num_stats['range'] = num_stats['max'] - num_stats['min']
print(num_stats.round(2))

print(f"\n购买转化率: {df['PurchaseStatus'].mean()*100:.2f}%")
print(f"总购买人数: {df['PurchaseStatus'].sum():,} 人")
print(f"总未购买人数: {(1-df['PurchaseStatus']).sum():,} 人")

print("\n分类型特征统计:")
cat_stats = df[['Gender', 'AgeGroup', 'IncomeLevel']].describe().T
print(cat_stats)

# 5. 数据预处理

# 创建年龄组
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[17, 30, 45, 60, 81],
    labels=['18-30岁', '31-45岁', '46-60岁', '60岁以上']
)

# 创建收入等级
income_quantiles = df['AnnualIncome'].quantile([0.25, 0.5, 0.75])
df['IncomeLevel'] = pd.cut(
    df['AnnualIncome'],
    bins=[0, income_quantiles[0.25], income_quantiles[0.5], income_quantiles[0.75], df['AnnualIncome'].max()+1],
    labels=['低收入', '中低收入', '中高收入', '高收入']
)

# 6. 单维度分析，分析不同单一维度对购买转化率的影响

print("\n" + "=" * 70)
print("三、单维度购买转化率分析")
print("=" * 70)

def analyze_dimension(df, group_col, target_col='PurchaseStatus', title=''):
    """维度分析函数"""
    result = df.groupby(group_col)[target_col].agg(['mean', 'count', 'sum'])
    result['转化率(%)'] = (result['mean'] * 100).round(2)
    result['购买人数'] = result['sum'].astype(int)
    result['未购买人数'] = (result['count'] - result['sum']).astype(int)
    return result[['转化率(%)', '购买人数', '未购买人数']]

# 6.1 年龄维度分析
print("\n【年龄维度】")
age_analysis = analyze_dimension(df, 'AgeGroup')
print(age_analysis)
max_age_idx = age_analysis['转化率(%)'].idxmax()
min_age_idx = age_analysis['转化率(%)'].idxmin()
print(f"\n结论: {max_age_idx} 转化率最高({age_analysis.loc[max_age_idx,'转化率(%)']}%), {min_age_idx} 转化率最低({age_analysis.loc[min_age_idx,'转化率(%)']}%)")

# 6.2 性别维度分析
print("\n【性别维度】")
gender_analysis = analyze_dimension(df, 'Gender')
print(gender_analysis)

# 6.3 收入维度分析
print("\n【收入维度】")
income_analysis = analyze_dimension(df, 'IncomeLevel')
print(income_analysis)
max_income_idx = income_analysis['转化率(%)'].idxmax()
print(f"\n结论: {max_income_idx} 人群转化率最高({income_analysis.loc[max_income_idx,'转化率(%)']}%)")

# 6.4 产品类别分析
print("\n【产品类别维度】")
category_analysis = analyze_dimension(df, 'ProductCategory')
category_analysis = category_analysis.sort_values('转化率(%)', ascending=False)
print(category_analysis)
top_category = category_analysis.index[0]
print(f"\n结论: {top_category} 转化率最高({category_analysis.iloc[0]['转化率(%)']}%), 最受消费者青睐")

# 6.5 客户满意度分析
print("\n【客户满意度维度】")
satisfaction_analysis = df.groupby('CustomerSatisfaction')['PurchaseStatus'].agg(['mean', 'count'])
satisfaction_analysis['转化率(%)'] = (satisfaction_analysis['mean'] * 100).round(2)
print(satisfaction_analysis[['转化率(%)']].round(2))


# 7. 交叉维度分析，从这里开始对不同维度的组合进行分析，以了解不同维度之间的关联关系
print("\n" + "=" * 70)
print("四、交叉维度分析")
print("=" * 70)

# 7.1 年龄 × 性别 交叉分析
print("\n【年龄 × 性别 交叉分析】")
cross_age_gender = pd.pivot_table(
    df, values='PurchaseStatus', index='AgeGroup', columns='Gender', aggfunc='mean'
) * 100
cross_age_gender = cross_age_gender.round(2)
print(cross_age_gender)

# 7.2 收入 × 产品类别 交叉分析
print("\n【收入等级 × 产品类别 交叉分析】")
cross_income_category = pd.pivot_table(
    df, values='PurchaseStatus', index='IncomeLevel', columns='ProductCategory', aggfunc='mean'
) * 100
cross_income_category = cross_income_category.round(2)
print(cross_income_category)

# 7.3 年龄 × 收入等级 交叉分析
print("\n【年龄 × 收入等级 交叉分析】")
cross_age_income = pd.pivot_table(
    df, values='PurchaseStatus', index='AgeGroup', columns='IncomeLevel', aggfunc='mean'
) * 100
cross_age_income = cross_age_income.round(2)
print(cross_age_income)

# 8. 统计检验，检验不同维度之间的关联性，判断是否有显著的关联
print("\n" + "=" * 70)
print("五、统计检验分析")
print("=" * 70)

alpha = 0.05

# 8.1 卡方检验 - 检验分类变量与购买行为的相关性
print("\n【卡方检验 - 分类变量关联性分析】")
chi2_results = []
for col in ['AgeGroup', 'Gender', 'IncomeLevel', 'ProductCategory']:
    contingency = pd.crosstab(df[col], df['PurchaseStatus'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    相关性 = "显著" if p_value < alpha else "不显著"
    chi2_results.append({
        '变量': col,
        '卡方值': round(chi2, 2),
        'p值': f"{p_value:.2e}",
        '自由度': dof,
        '结论': f"p<0.05={相关性}, {col}与购买行为{'存在' if p_value < alpha else '不存在'}显著关联"
    })
    print(f"  {col}: χ²={chi2:.2f}, p={p_value:.2e} → {相关性}关联")

# 8.2 相关性分析
print("\n【皮尔逊相关性分析】")
numeric_cols = ['Age', 'AnnualIncome', 'CustomerSatisfaction', 'PurchaseStatus']
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix.round(3))

# 9. 可视化分析，展示不同维度的转化率分布

print("\n" + "=" * 70)
print("六、可视化分析")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 9.1 年龄组转化率
ax1 = axes[0, 0]
age_conv = df.groupby('AgeGroup')['PurchaseStatus'].mean() * 100
colors = sns.color_palette("viridis", len(age_conv))
bars = ax1.bar(age_conv.index.astype(str), age_conv.values, color=colors)
ax1.set_title('不同年龄组购买转化率', fontsize=12, fontweight='bold')
ax1.set_xlabel('年龄组')
ax1.set_ylabel('转化率 (%)')
for bar, val in zip(bars, age_conv.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=10)

# 9.2 性别转化率
ax2 = axes[0, 1]
gender_conv = df.groupby('Gender')['PurchaseStatus'].mean() * 100
bars = ax2.bar(gender_conv.index, gender_conv.values, color=['#3498db', '#e74c3c'])
ax2.set_title('不同性别购买转化率', fontsize=12, fontweight='bold')
ax2.set_xlabel('性别')
ax2.set_ylabel('转化率 (%)')
for bar, val in zip(bars, gender_conv.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=10)

# 9.3 产品类别转化率
ax3 = axes[0, 2]
cat_conv = df.groupby('ProductCategory')['PurchaseStatus'].mean() * 100
cat_conv = cat_conv.sort_values(ascending=True)
bars = ax3.barh(cat_conv.index, cat_conv.values, color=sns.color_palette("plasma", len(cat_conv)))
ax3.set_title('不同产品类别购买转化率', fontsize=12, fontweight='bold')
ax3.set_xlabel('转化率 (%)')
for bar, val in zip(bars, cat_conv.values):
    ax3.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=9)

# 9.4 满意度与转化率关系
ax4 = axes[1, 0]
satisfaction_conv = df.groupby('CustomerSatisfaction')['PurchaseStatus'].mean() * 100
ax4.plot(satisfaction_conv.index, satisfaction_conv.values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax4.fill_between(satisfaction_conv.index, satisfaction_conv.values, alpha=0.3, color='#2E86AB')
ax4.set_title('客户满意度与购买转化率关系', fontsize=12, fontweight='bold')
ax4.set_xlabel('满意度评分 (1-5)')
ax4.set_ylabel('转化率 (%)')
ax4.set_xticks(range(1, 6))

# 9.5 收入分布箱线图
ax5 = axes[1, 1]
df.boxplot(column='AnnualIncome', by='PurchaseStatus', ax=ax5)
ax5.set_title('购买状态收入分布', fontsize=12, fontweight='bold')
ax5.set_xlabel('购买状态 (0=未购买, 1=购买)')
ax5.set_ylabel('年收入')
plt.suptitle('')

# 9.6 相关性热力图
ax6 = axes[1, 2]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, ax=ax6, fmt='.2f',
            square=True, linewidths=0.5)
ax6.set_title('特征相关性热力图', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'analysis_dashboard.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\n可视化仪表板已保存至: analysis_dashboard.png")

# 10. 购买预测模型，使用逻辑回归模型对购买行为进行预测
print("\n" + "=" * 70)
print("七、购买预测模型 (逻辑回归)")
print("=" * 70)

# 特征编码
df_model = df.copy()
df_model['Gender_encoded'] = pd.get_dummies(df_model['Gender'], drop_first=True)
df_model = pd.get_dummies(df_model, columns=['AgeGroup', 'IncomeLevel', 'ProductCategory'], drop_first=True)

# 特征选择
feature_cols = [col for col in df_model.columns if col not in ['PurchaseStatus', 'Gender']]
X = df_model[feature_cols]
y = df_model['PurchaseStatus']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n训练集: {len(X_train):,} 样本, 测试集: {len(X_test):,} 样本")

# 模型训练
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"\n模型准确率: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"AUC-ROC得分: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n分类报告:")
target_names = ['未购买', '购买']
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(f"  预测→    未购买   购买")
print(f"  实际↓")
print(f"  未购买    {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"  购买      {cm[1,0]:5d}  {cm[1,1]:5d}")

# 特征重要性 (逻辑回归系数)
print("\n特征重要性 (Top 10):")
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '系数': model.coef_[0]
}).sort_values('系数', key=abs, ascending=False)
print(feature_importance.head(10).to_string(index=False))


# 11. 分析结论总结

print("\n" + "=" * 70)
print("八、分析结论与业务建议")
print("=" * 70)

conclusions = """
【核心发现】

1. 整体转化率
   - 整体购买转化率为 {:.2f}%，处于行业正常水平
   - 总样本中购买人数 {:,} 人，未购买 {:,} 人

2. 年龄维度
   - 转化率最高的年龄段: {} ({:.2f}%)
   - 转化率最低的年龄段: {} ({:.2f}%)
   - 建议: 重点关注高转化年龄段的营销策略，同时研究低转化年龄段的需求差异

3. 性别维度
   - {} 转化率 ({:.2f}%) 略高于 {} ({:.2f}%)
   - 性别对购买决策的影响需结合其他因素综合分析

4. 收入维度
   - {} 群体转化率最高 ({:.2f}%)
   - 收入水平与购买意愿呈现{}关系

5. 产品类别
   - 最受欢迎类别: {} (转化率 {:.2f}%)
   - 最不受欢迎类别: {} (转化率 {:.2f}%)
   - 建议: 加大对高转化率类别的库存和营销投入

6. 客户满意度
   - 满意度与购买转化率呈{}相关
   - 满意度每增加1分，转化率平均{}约{:.1f}个百分点
   - 建议: 持续优化客户服务，提升客户满意度

7. 模型预测
   - 逻辑回归模型准确率达到 {:.2f}%
   - AUC-ROC得分为 {:.4f}，模型具有{}
   - 可用于潜在客户识别和精准营销

【业务建议】

1. 精准营销: 针对{}年龄段和{}收入群体重点推广
2. 产品策略: 加大{}类产品的营销力度
3. 客户维系: 关注客户满意度，及时处理客户反馈
4. 资源分配: 基于预测模型识别高意向客户，优化广告投放
""".format(
    df['PurchaseStatus'].mean() * 100,
    df['PurchaseStatus'].sum(),
    (1 - df['PurchaseStatus']).sum(),
    age_analysis['转化率(%)'].idxmax(), age_analysis['转化率(%)'].max(),
    age_analysis['转化率(%)'].idxmin(), age_analysis['转化率(%)'].min(),
    gender_analysis.index[gender_analysis['转化率(%)'].argmax()],
    gender_analysis['转化率(%)'].max(),
    gender_analysis.index[gender_analysis['转化率(%)'].argmin()],
    gender_analysis['转化率(%)'].min(),
    income_analysis['转化率(%)'].idxmax(), income_analysis['转化率(%)'].max(),
    '正' if satisfaction_conv.corr(pd.Series(range(1,6))) > 0 else '负',
    '上升' if satisfaction_conv.diff().mean() > 0 else '下降',
    abs(satisfaction_conv.diff().mean()),
    accuracy_score(y_test, y_pred) * 100,
    roc_auc_score(y_test, y_pred_proba),
    '良好的区分能力' if roc_auc_score(y_test, y_pred_proba) > 0.7 else '有限的区分能力',
    age_analysis['转化率(%)'].idxmax().replace('岁', ''),
    income_analysis['转化率(%)'].idxmax(),
    category_analysis.index[0]
)

print(conclusions)

print("=" * 70)
print("分析完成")
print("=" * 70)