"""
电商用户聚类分析 - K-Means算法实现

本模块使用真实电商用户数据，基于用户行为（购买状态、满意度）和
人口统计学特征（年龄、收入）进行综合聚类分析，实现用户分群。

数据来源: customerData_500k.csv
聚类特征: PurchaseStatus, CustomerSatisfaction, Age, AnnualIncome

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import time
import os

warnings.filterwarnings('ignore')

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 前期准备
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# 加载数据并初步探索

def load_data(file_path):

    df = pd.read_csv(file_path)
    print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def explore_data(df):

    explore_info = {}

    # 数据基本信息
    print("\n【数据基本信息】")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1]}")
    print(f"\n列名: {df.columns.tolist()}")

    # 缺失值检查
    print("\n【缺失值检查】")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ 数据无缺失值")
    else:
        print(missing[missing > 0])

    # 重复值检查
    print("\n【重复值检查】")
    duplicates = df.duplicated().sum()
    print(f"重复记录: {duplicates} 条")

    # 数值特征统计
    print("\n【数值特征统计】")
    numeric_cols = ['Age', 'AnnualIncome', 'PurchaseStatus', 'CustomerSatisfaction']
    stats = df[numeric_cols].describe().T
    stats['range'] = stats['max'] - stats['min']
    print(stats.round(2))

    # 购买转化率
    print("\n【购买行为统计】")
    purchase_rate = df['PurchaseStatus'].mean() * 100
    print(f"购买转化率: {purchase_rate:.2f}%")
    print(f"购买人数: {df['PurchaseStatus'].sum():,} 人")
    print(f"未购买人数: {(1 - df['PurchaseStatus']).sum():,} 人")

    # 保存探索信息
    explore_info['sample_count'] = df.shape[0]
    explore_info['purchase_rate'] = purchase_rate
    explore_info['numeric_stats'] = stats

    return explore_info

# 数据预处理模块

def preprocess_for_clustering(df, feature_cols):

    # 选取特征
    X = df[feature_cols].copy()
    print(f"\n选取聚类特征: {feature_cols}")

    # 处理缺失值（使用中位数填充）
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"发现 {missing_count} 个缺失值，使用中位数填充")
        X = X.fillna(X.median())

    # 处理异常值（使用IQR方法裁剪）
    print("\n异常值处理 (IQR 1.5倍范围):")
    for col in feature_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((X[col] < lower) | (X[col] > upper)).sum()
        if outliers > 0:
            print(f"  {col}: {outliers} 个异常值已裁剪")
            X[col] = X[col].clip(lower, upper)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n✓ 数据标准化完成 (均值=0, 标准差=1)")

    # 保存清洗后的数据
    df_clean = df.copy()
    df_clean[feature_cols] = X

    return X_scaled, scaler, df_clean

# K-Means模型模块

def train_kmeans(X, n_clusters=4, random_state=42):

    start_time = time.time()
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X)
    train_time = time.time() - start_time

    print(f"\n✓ K-Means模型训练完成 (K={n_clusters})")
    print(f"  训练耗时: {train_time:.4f} 秒")
    print(f"  惯性值 (Inertia): {kmeans.inertia_:.2f}")

    return kmeans, train_time

# 模型评估模块

def evaluate_clustering(X, labels, kmeans):

    print("\n" + "=" * 60)
    print("聚类效果评估")
    print("=" * 60)

    metrics = {}

    # 轮廓系数
    silhouette = silhouette_score(X, labels)
    metrics['silhouette'] = silhouette
    print(f"\n轮廓系数 (Silhouette Score): {silhouette:.4f}")
    print("  解读: 范围[-1,1]，越接近1表示聚类效果越好")

    # 各簇样本数量
    print("\n各簇样本分布:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        print(f"  簇 {cluster}: {count:,} 人 ({percentage:.1f}%)")

    metrics['cluster_counts'] = dict(zip(unique, counts))

    # 簇内方差
    print(f"\n簇内方差 (Inertia): {kmeans.inertia_:.2f}")

    return metrics


def find_optimal_k(X, k_range=None):

    if k_range is None:
        k_range = range(2, 11)

    print("\n" + "=" * 60)
    print("最佳聚类数量分析")
    print("=" * 60)

    inertias = []
    silhouettes = []
    k_range_list = list(k_range)

    print("\n遍历K值进行评估...")
    for k in k_range_list:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette = silhouette_score(X, kmeans.labels_)
        silhouettes.append(silhouette)
        print(f"  K={k}: 惯性值={kmeans.inertia_:.2f}, 轮廓系数={silhouette:.4f}")

    # 绘制评估图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 肘部法则图
    axes[0].plot(k_range_list, inertias, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_title('Elbow Method (肘部法则)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia (惯性值)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 轮廓系数图
    axes[1].plot(k_range_list, silhouettes, 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[1].set_title('Silhouette Score (轮廓系数法)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_k_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 选择最佳K值（轮廓系数最高）
    best_k_idx = np.argmax(silhouettes)
    best_k = k_range_list[best_k_idx]
    best_silhouette = silhouettes[best_k_idx]

    print(f"\n根据轮廓系数分析，最佳聚类数量: K = {best_k}")
    print(f"对应的轮廓系数: {best_silhouette:.4f}")

    results = {
        'k_range': k_range_list,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'best_k': best_k,
        'best_silhouette': best_silhouette
    }

    return results

# 可视化模块

def visualize_clusters_2d(X, labels, centers, feature_names, title="用户聚类结果"):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 选择前两个特征进行可视化
    feat1, feat2 = feature_names[0], feature_names[1]

    # 定义簇的颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # 左图：原始数据聚类结果
    for cluster in np.unique(labels):
        mask = labels == cluster
        axes[0].scatter(
            X[mask, 0], X[mask, 1],
            c=colors[cluster % len(colors)],
            label=f'Cluster {cluster} (n={mask.sum()})',
            alpha=0.6,
            s=50
        )

    axes[0].scatter(
        centers[:, 0], centers[:, 1],
        c='black',
        marker='X',
        s=200,
        linewidths=3,
        label='Cluster Centers'
    )
    axes[0].set_xlabel(feat1, fontsize=12)
    axes[0].set_ylabel(feat2, fontsize=12)
    axes[0].set_title(f'{title}\n({feat1} vs {feat2})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右图：购买状态 vs 满意度
    purchase_mask = X[:, 2] == 1  # PurchaseStatus
    non_purchase_mask = X[:, 2] == 0

    for i, (mask, label, color) in enumerate([
        (purchase_mask, 'Purchased (购买)', '#2ECC71'),
        (non_purchase_mask, 'Not Purchased (未购买)', '#E74C3C')
    ]):
        axes[1].scatter(
            X[mask, 3],  # CustomerSatisfaction
            X[mask, 1],  # AnnualIncome
            c=color,
            label=label,
            alpha=0.5,
            s=30
        )

    axes[1].set_xlabel('Customer Satisfaction (满意度)', fontsize=12)
    axes[1].set_ylabel('Annual Income (年收入)', fontsize=12)
    axes[1].set_title('Purchase Behavior vs Satisfaction', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()


def visualize_cluster_profiles(df, labels, feature_cols):

    df_plot = df.copy()
    df_plot['Cluster'] = labels

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 各簇年龄分布箱线图
    sns.boxplot(data=df_plot, x='Cluster', y='Age', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Age Distribution by Cluster', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Cluster', fontsize=11)
    axes[0, 0].set_ylabel('Age', fontsize=11)

    # 2. 各簇收入分布箱线图
    sns.boxplot(data=df_plot, x='Cluster', y='AnnualIncome', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Income Distribution by Cluster', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Cluster', fontsize=11)
    axes[0, 1].set_ylabel('Annual Income', fontsize=11)

    # 3. 各簇购买转化率
    purchase_rate = df_plot.groupby('Cluster')['PurchaseStatus'].mean() * 100
    bars = axes[1, 0].bar(purchase_rate.index, purchase_rate.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 0].set_title('Purchase Rate by Cluster', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Cluster', fontsize=11)
    axes[1, 0].set_ylabel('Purchase Rate (%)', fontsize=11)
    for bar, val in zip(bars, purchase_rate.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', fontsize=10)

    # 4. 各簇满意度分布箱线图
    sns.boxplot(data=df_plot, x='Cluster', y='CustomerSatisfaction', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Satisfaction Distribution by Cluster', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Cluster', fontsize=11)
    axes[1, 1].set_ylabel('Customer Satisfaction', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_profiles.png'), dpi=150, bbox_inches='tight')
    plt.show()


def visualize_cluster_summary(df, labels, feature_cols):

    df_plot = df.copy()
    df_plot['Cluster'] = labels

    # 计算各簇特征均值
    cluster_means = df_plot.groupby('Cluster')[feature_cols].mean()

    # 归一化到0-1范围便于可视化
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建雷达图
    categories = ['Age', 'Annual Income', 'Purchase Status', 'Satisfaction']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for idx, (cluster, row) in enumerate(cluster_means_norm.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}', color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('Cluster Feature Profile (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_radar.png'), dpi=150, bbox_inches='tight')
    plt.show()

# 用户分群分析模块

def analyze_clusters(df, labels, feature_cols):

    print("\n" + "=" * 60)
    print("用户分群分析报告")
    print("=" * 60)

    df_analysis = df.copy()
    df_analysis['Cluster'] = labels

    # 按簇分组统计
    cluster_stats = df_analysis.groupby('Cluster').agg({
        'Age': ['mean', 'min', 'max'],
        'AnnualIncome': ['mean', 'min', 'max'],
        'PurchaseStatus': ['mean', 'sum'],
        'CustomerSatisfaction': ['mean', 'min', 'max']
    }).round(2)

    cluster_profiles = {}

    for cluster in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        n_users = len(cluster_data)

        profile = {
            'count': n_users,
            'percentage': n_users / len(df_analysis) * 100,
            'avg_age': cluster_data['Age'].mean(),
            'avg_income': cluster_data['AnnualIncome'].mean(),
            'purchase_rate': cluster_data['PurchaseStatus'].mean() * 100,
            'avg_satisfaction': cluster_data['CustomerSatisfaction'].mean()
        }

        cluster_profiles[cluster] = profile

        # 打印簇详情
        print(f"\n【簇 {cluster}】({n_users:,} 人, {profile['percentage']:.1f}%)")
        print("-" * 40)
        print(f"  平均年龄: {profile['avg_age']:.1f} 岁")
        print(f"  平均年收入: ${profile['avg_income']:,.0f}")
        print(f"  购买转化率: {profile['purchase_rate']:.1f}%")
        print(f"  平均满意度: {profile['avg_satisfaction']:.2f}/5.0")

    # 簇间对比
    print("\n" + "=" * 60)
    print("簇间对比总结")
    print("=" * 60)

    summary_df = pd.DataFrame(cluster_profiles).T
    summary_df.columns = ['用户数', '占比(%)', '平均年龄', '平均收入', '购买率(%)', '满意度']
    print(summary_df.round(2).to_string())

    return cluster_profiles

# 主程序入口

def main():
    """
    主函数：执行完整的电商用户聚类分析流程

    流程:
    1. 加载数据
    2. 数据探索
    3. 数据预处理
    4. 寻找最佳K值
    5. 训练K-Means模型
    6. 评估聚类效果
    7. 可视化聚类结果
    8. 用户分群分析
    """
    plt.ion()  # 开启交互模式，避免plt.show()阻塞
    print("\n" + "=" * 60)
    print("  电商用户聚类分析系统")
    print("  K-Means Clustering Analysis")
    print("=" * 60)

    # 配置参数
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'e-commerce-purchase-analysis', 'customerData_500k.csv')
    FEATURE_COLS = ['Age', 'AnnualIncome', 'PurchaseStatus', 'CustomerSatisfaction']
    INITIAL_K = 4
    SAMPLE_SIZE = 50000  # 采样5万条数据加速处理

    # 第一步：加载数据
    print("\n>>> 第一步：加载数据")
    df = load_data(DATA_PATH)

    # 对大数据集进行采样以加速处理
    if len(df) > SAMPLE_SIZE:
        print(f"\n原始数据量: {len(df):,}, 采样: {SAMPLE_SIZE:,}")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"采样完成: {len(df):,}")

    # 第二步：数据探索
    print("\n>>> 第二步：数据探索")
    explore_info = explore_data(df)

    # 第三步：数据预处理
    print("\n>>> 第三步：数据预处理")
    X_scaled, scaler, df_clean = preprocess_for_clustering(df, FEATURE_COLS)

    # 第四步：寻找最佳K值
    print("\n>>> 第四步：寻找最佳聚类数量")
    optimal_results = find_optimal_k(X_scaled)
    optimal_k = optimal_results['best_k']

    # 第五步：训练最终模型
    print("\n>>> 第五步：训练K-Means模型")
    final_k = optimal_k if optimal_k != INITIAL_K else INITIAL_K
    print(f"使用聚类数量: K = {final_k}")

    kmeans, train_time = train_kmeans(X_scaled, n_clusters=final_k)
    cluster_labels = kmeans.labels_
    cluster_centers_scaled = kmeans.cluster_centers_

    # 第六步：评估聚类效果
    print("\n>>> 第六步：评估聚类效果")
    metrics = evaluate_clustering(X_scaled, cluster_labels, kmeans)

    # 第七步：可视化聚类结果
    print("\n>>> 第七步：可视化聚类结果")

    # 获取原始尺度的中心点坐标用于可视化
    cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

    visualize_clusters_2d(
        df_clean[FEATURE_COLS].values,
        cluster_labels,
        cluster_centers_original,
        FEATURE_COLS
    )

    visualize_cluster_profiles(df_clean, cluster_labels, FEATURE_COLS)
    visualize_cluster_summary(df_clean, cluster_labels, FEATURE_COLS)

    # 第八步：用户分群分析
    print("\n>>> 第八步：用户分群分析")
    cluster_profiles = analyze_clusters(df_clean, cluster_labels, FEATURE_COLS)

    # 保存聚类结果
    print("\n>>> 保存聚类结果")
    df_result = df.copy()
    df_result['Cluster'] = cluster_labels
    df_result.to_csv(os.path.join(OUTPUT_DIR, 'user_clustering_results.csv'), index=False)
    print("✓ 聚类结果已保存至 user_clustering_results.csv")

    print("\n" + "=" * 60)
    print("  分析完成！")
    print("=" * 60)

    return df_result, cluster_profiles


if __name__ == "__main__":
    df_result, profiles = main()
