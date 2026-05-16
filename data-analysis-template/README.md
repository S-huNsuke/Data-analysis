# 交互式数据分析模板

## 项目简介
基于 Rich 库的交互式终端数据分析工具，支持数据加载、清洗、探索性分析、可视化和机器学习，自动生成 Markdown 分析报告。

## 功能特性

### 数据加载
- 支持 CSV / Excel 文件加载
- 自动检测文件编码
- 数据预览与基本信息展示

### 数据清洗
- 缺失值统计与处理（均值/中位数/众数/前向填充/后向填充/删除）
- 异常值检测（IQR 方法 / Z-score 方法）
- 重复记录删除
- 数据类型转换

### 探索性分析
- 描述性统计
- 相关性分析
- 分组统计
- 分布分析（偏度、峰度等）

### 可视化图表
- 折线图、柱状图、饼图
- 热力图、箱线图、散点图
- 直方图

### 机器学习
- 逻辑回归分类
- 随机森林分类
- K-Means 聚类
- 线性回归

### 报告生成
- 自动生成 Markdown 格式分析报告
- 包含数据概览、统计摘要、图表路径、分析结论

## 技术栈
- **TUI 框架**: Rich
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **机器学习**: scikit-learn

## 项目结构
```
data-analysis-template/
├── main.py                # 交互式 TUI 主程序
├── modules/
│   ├── loader.py          # 数据加载模块
│   ├── cleaner.py         # 数据清洗模块
│   ├── eda.py             # 探索性分析模块
│   ├── visualizer.py      # 可视化模块
│   └── modeler.py         # 机器学习模块
├── reports/
│   └── generator.py       # 报告生成模块
├── utils/
│   └── helpers.py         # 工具函数
└── output/
    ├── reports/           # 分析报告
    └── charts/            # 可视化图表
```

## 使用方法

```bash
# 从项目根目录运行
uv run python data-analysis-template/main.py
```

启动后按菜单提示操作：
1. 选择「数据加载」输入文件路径
2. 选择分析模块执行分析
3. 选择「生成报告」导出结果

输出结果保存在 `output/` 目录下。
