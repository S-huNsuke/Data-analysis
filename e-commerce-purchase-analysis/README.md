# 电商购买行为分析及建模

## 项目简介
基于用户人口统计特征和购买行为数据，进行购买转化率分析和逻辑回归预测建模

## 功能特性
- 数据质量检测
- 描述性统计
- 单维度/交叉维度分析
- 统计检验（卡方/皮尔逊）
- 可视化仪表板
- 逻辑回归预测模型

## 技术栈
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## 项目结构
```
e-commerce-purchase-analysis/
├── main.py              # 入口脚本
├── customerData_500k.csv # 数据集
└── output/              # 输出结果
```

## 使用方法

```bash
# 从项目根目录运行
uv run python e-commerce-purchase-analysis/main.py
```

输出结果保存在 `output/` 目录下。
