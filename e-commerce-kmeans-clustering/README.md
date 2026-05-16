# 电商用户K-means聚类分析

## 项目简介
基于用户行为和人口统计特征，使用K-Means算法进行用户分群

## 功能特性
- 数据探索与预处理
- 最佳聚类数量分析（肘部法则/轮廓系数法）
- K-Means聚类
- 聚类效果评估
- 2D可视化
- 簇特征画像
- 雷达图
- 用户分群分析报告

## 技术栈
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 项目结构
```
e-commerce-kmeans-clustering/
├── main.py    # 入口脚本
└── output/    # 输出结果
```

## 使用方法

```bash
# 从项目根目录运行
uv run python e-commerce-kmeans-clustering/main.py
```

输出结果保存在 `output/` 目录下。

> **注意**：数据文件引用自 `e-commerce-purchase-analysis/customerData_500k.csv`，请确保该文件存在。
