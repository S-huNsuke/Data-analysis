# 滴滴出行AB测试

## 项目简介
基于滴滴出行实验数据，进行假设检验和A/B测试分析

## 功能特性
- 特征工程（ROI/转化率/取消率）
- 正态性检验（Shapiro-Wilk）
- 方差齐性检验（Levene）
- 独立样本t检验
- 非参数检验（Mann-Whitney U）
- 可视化对比分析

## 技术栈
- pandas
- numpy
- matplotlib
- scipy

## 项目结构
```
didi-ab-testing/
├── main.py     # 入口脚本
└── test.xlsx   # 数据集（需自行准备）
```

## 使用方法

```bash
# 从项目根目录运行
uv run python didi-ab-testing/main.py
```

> **注意**：数据文件 `test.xlsx` 需自行准备并放置于项目目录下。
