# 泰坦尼克号生存预测

## 项目简介
经典数据挖掘全流程实战，从探索性分析到多模型对比与集成学习

## 功能特性
- 探索性数据分析
- 特征工程（称谓提取/年龄填充/家庭规模）
- 多模型训练对比（SVM/KNN/决策树/随机森林/朴素贝叶斯/逻辑回归）
- 交叉验证
- 超参数网格搜索
- 集成学习（投票/Bagging/AdaBoost/Gradient Boosting/XGBoost）
- 特征重要性分析

## 技术栈
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## 项目结构
```
titanic-survival-prediction/
├── main.py    # 入口脚本
├── train.csv  # 训练数据集
├── test.csv   # 测试数据集
└── output/    # 输出结果
```

## 使用方法

```bash
# 从项目根目录运行
uv run python titanic-survival-prediction/main.py
```

输出结果保存在 `output/` 目录下。
