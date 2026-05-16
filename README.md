# Data Analysis

基于 Python 的数据分析项目集合。

## 项目列表

### 数据分析项目

| 项目 | 目录 | 说明 |
|------|------|------|
| 电商购买行为分析及建模 | `e-commerce-purchase-analysis/` | 用户购买转化率分析、逻辑回归预测模型 |
| 天气数据可视化 | `weather-visualization/` | 24小时/7天温度、湿度、风力趋势分析与可视化 |
| B站UP主数据可视化 | `bilibili-up-analysis/` | B站榜单数据分析、词云生成、树状图可视化 |
| 游戏销售数据分析 | `game-sales-analysis/` | 全球游戏销售数据统计与可视化 |
| 泰坦尼克号生存预测 | `titanic-survival-prediction/` | 数据挖掘全流程实战，多模型对比与集成学习 |
| 滴滴出行AB测试 | `didi-ab-testing/` | 假设检验、正态性检验、A/B测试分析 |
| 电商用户K-means聚类 | `e-commerce-kmeans-clustering/` | 基于用户行为和人口统计特征的K-Means聚类分群 |
| 电商用户特征分析 | `e-commerce-user-analysis/` | 用户个性化特征分析、RFM分群 |

### 工具项目

| 项目 | 目录 | 说明 |
|------|------|------|
| 交互式数据分析模板 | `data-analysis-template/` | 交互式 TUI 数据分析工具，支持加载、清洗、分析、可视化、建模 |

## 项目结构

```
Data-analysis/
├── e-commerce-purchase-analysis/      # 电商购买行为分析
│   ├── main.py                        # 入口脚本
│   ├── customerData_500k.csv          # 数据文件
│   └── output/                        # 输出结果
├── weather-visualization/             # 天气数据可视化
│   ├── get_data.py                    # 数据采集
│   ├── hourly_plt.py                  # 24小时可视化
│   ├── week_plt.py                    # 7天可视化
│   ├── *.csv                          # 数据文件
│   └── output/                        # 输出结果
├── bilibili-up-analysis/              # B站UP主分析
│   ├── main.py                        # 入口脚本
│   ├── *.xlsx / *.txt / *.jpg        # 数据与资源文件
│   └── output/                        # 输出结果
├── game-sales-analysis/               # 游戏销售分析
│   ├── main.py                        # 入口脚本
│   ├── vgsales.csv                    # 数据文件
│   └── output/                        # 输出结果
├── titanic-survival-prediction/       # 泰坦尼克号生存预测
│   ├── main.py                        # 入口脚本
│   ├── train.csv / test.csv           # 数据文件
│   └── output/                        # 输出结果
├── didi-ab-testing/                   # 滴滴AB测试
│   └── main.py                        # 入口脚本
├── e-commerce-kmeans-clustering/      # 电商用户聚类
│   ├── main.py                        # 入口脚本
│   └── output/                        # 输出结果
├── e-commerce-user-analysis/          # 电商用户特征分析
│   ├── main.py                        # 入口脚本
│   └── user_personalized_features.csv # 数据文件
├── data-analysis-template/            # 交互式数据分析模板
│   ├── main.py                        # TUI 主程序
│   ├── modules/                       # 分析模块
│   ├── reports/                       # 报告生成
│   ├── utils/                         # 工具函数
│   └── output/                        # 输出结果
├── pyproject.toml                     # 项目依赖（uv 管理）
├── .python-version                    # Python 版本
└── .gitignore
```

## 技术栈

- **语言**: Python 3.12+
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn, pyecharts
- **机器学习**: scikit-learn, xgboost
- **爬虫**: crawl4ai, beautifulsoup4
- **NLP**: jieba, wordcloud
- **TUI**: Rich

## 使用方法

1. 安装依赖：
```bash
uv sync
```

2. 运行子项目：
```bash
uv run python e-commerce-purchase-analysis/main.py
uv run python weather-visualization/get_data.py
uv run python bilibili-up-analysis/main.py
uv run python game-sales-analysis/main.py
uv run python titanic-survival-prediction/main.py
uv run python didi-ab-testing/main.py
uv run python e-commerce-kmeans-clustering/main.py
uv run python e-commerce-user-analysis/main.py
```

3. 运行交互式数据分析模板：
```bash
uv run python data-analysis-template/main.py
```

各子项目的输出结果保存在对应的 `output/` 目录下。
