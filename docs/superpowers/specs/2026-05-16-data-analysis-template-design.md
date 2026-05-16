# 交互式数据分析模板 - 设计文档

**日期**: 2026-05-16
**状态**: 已确认

---

## 1. 项目概述

创建一个基于 Rich 库的交互式 TUI（终端用户界面）数据分析模板。用户上传数据文件后，通过菜单选择要执行的分析模块，系统自动执行分析并生成 Markdown 报告和可视化图表。

### 核心特性

- 交互式 TUI 界面，基于 Rich 库
- 支持 CSV、Excel 等结构化数据
- 模块化设计，可独立扩展
- 自动生成分析报告（Markdown）和可视化图表

---

## 2. 系统架构

```
data-analysis-template/
├── main.py                    # 主入口，交互式调度
├── modules/
│   ├── __init__.py
│   ├── loader.py              # 数据加载模块
│   ├── cleaner.py             # 数据清洗模块
│   ├── eda.py                 # 探索性分析模块
│   ├── visualizer.py          # 可视化图表模块
│   └── modeler.py             # 机器学习模块
├── reports/
│   └── generator.py           # Markdown 报告生成器
├── utils/
│   └── helpers.py             # 通用工具函数
└── output/
    ├── reports/                # 分析报告输出
    └── charts/                 # 可视化图表输出
```

---

## 3. 目录结构规范

| 目录/文件 | 说明 |
|----------|------|
| `main.py` | 主程序入口，TUI 交互界面 |
| `modules/` | 分析模块包 |
| `modules/loader.py` | 数据加载，支持 CSV/Excel |
| `modules/cleaner.py` | 数据清洗模块 |
| `modules/eda.py` | 探索性数据分析模块 |
| `modules/visualizer.py` | 可视化图表模块 |
| `modules/modeler.py` | 机器学习模块 |
| `reports/generator.py` | Markdown 报告生成 |
| `utils/helpers.py` | 工具函数 |
| `output/` | 输出目录 |

---

## 4. 模块详细设计

### 4.1 数据加载模块 (loader.py)

**功能**:
- 支持文件格式: CSV, Excel (.xlsx, .xls)
- 自动检测分隔符、编码
- 返回 pandas DataFrame 和基本信息

**接口**:
```python
def load_data(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    加载数据文件
    返回: (数据框, 基本信息字典)
    """
```

### 4.2 数据清洗模块 (cleaner.py)

**功能**:
| 功能 | 说明 |
|------|------|
| 缺失值处理 | 删除/填充（均值/中位数/众数/指定值） |
| 异常值检测 | IQR 方法、Z-score 方法 |
| 数据类型转换 | 自动推断、手动指定 |
| 重复值删除 | 检测并删除重复记录 |
| 字符串处理 | 去除空格、统一大小写 |

**接口**:
```python
def get_data_info(df: pd.DataFrame) -> dict:
    """获取数据基本信息"""

def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """处理缺失值"""

def detect_outliers(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """检测异常值"""

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """删除重复记录"""

def convert_dtype(df: pd.DataFrame, col: str, dtype: str) -> pd.DataFrame:
    """数据类型转换"""
```

### 4.3 探索性分析模块 (eda.py)

**功能**:
| 功能 | 说明 |
|------|------|
| 描述性统计 | 均值、中位数、标准差、分位数等 |
| 分布分析 | 直方图分布、偏度、峰度 |
| 相关性分析 | 相关系数矩阵、热力图 |
| 分组统计 | 按列分组聚合分析 |
| 交叉分析 | 两个维度的交叉统计 |

**接口**:
```python
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """描述性统计"""

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """相关性分析"""

def group_analysis(df: pd.DataFrame, group_col: str, agg_col: str) -> pd.DataFrame:
    """分组统计"""

def distribution_analysis(df: pd.DataFrame, col: str) -> dict:
    """分布分析"""
```

### 4.4 可视化模块 (visualizer.py)

**功能**:
| 图表类型 | 适用场景 |
|---------|---------|
| 折线图 | 时间序列、趋势变化 |
| 柱状图 | 分类对比、数量统计 |
| 饼图 | 占比分析、比例展示 |
| 热力图 | 相关性矩阵、缺失值分布 |
| 箱线图 | 分布情况、异常值检测 |
| 散点图 | 两个变量关系分析 |
| 直方图 | 单变量分布分析 |
| 堆叠图 | 多类别占比变化 |

**接口**:
```python
def plot_line(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    """折线图，返回保存路径"""

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    """柱状图，返回保存路径"""

def plot_pie(df: pd.DataFrame, col: str, title: str) -> str:
    """饼图，返回保存路径"""

def plot_heatmap(df: pd.DataFrame, title: str) -> str:
    """热力图，返回保存路径"""

def plot_box(df: pd.DataFrame, col: str, title: str) -> str:
    """箱线图，返回保存路径"""

def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    """散点图，返回保存路径"""

def plot_histogram(df: pd.DataFrame, col: str, title: str) -> str:
    """直方图，返回保存路径"""
```

### 4.5 机器学习模块 (modeler.py)

**功能**:

| 任务类型 | 算法 | 说明 |
|---------|------|------|
| 分类 | 逻辑回归 | 二分类问题 |
| 分类 | 随机森林 | 多分类/二分类，特征重要性 |
| 聚类 | K-Means | 用户分群、样本聚类 |
| 回归 | 线性回归 | 数值预测 |

**接口**:
```python
def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """逻辑回归分类"""

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> dict:
    """随机森林分类"""

def train_kmeans(X: pd.DataFrame, n_clusters: int) -> dict:
    """K-Means 聚类"""

def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """线性回归"""
```

### 4.6 报告生成模块 (reports/generator.py)

**功能**:
- 生成 Markdown 格式分析报告
- 包含数据概览、统计结果、分析结论
- 自动嵌入图表路径
- 支持自定义报告标题和描述

**接口**:
```python
def generate_report(
    title: str,
    data_info: dict,
    stats: dict,
    charts: list[str],
    output_path: str
) -> str:
    """生成 Markdown 报告，返回报告路径"""
```

---

## 5. 交互流程设计

### 5.1 主菜单结构

```
╔══════════════════════════════════════════════════════════╗
║            交互式数据分析模板 v1.0                       ║
╠══════════════════════════════════════════════════════════╣
║  [1] 数据加载         - 加载 CSV/Excel 文件              ║
║  [2] 数据清洗         - 缺失值、异常值处理              ║
║  [3] 探索性分析       - 描述统计、相关性分析            ║
║  [4] 可视化图表       - 生成各类统计图表                ║
║  [5] 机器学习         - 分类、聚类、回归                ║
║  [6] 生成报告         - 导出 Markdown 分析报告          ║
║  [0] 退出             - 退出程序                        ║
╚══════════════════════════════════════════════════════════╝
```

### 5.2 交互流程

```
1. 启动程序 → 显示主菜单
2. 选择 [1] 数据加载 → 输入文件路径 → 显示数据预览
3. 返回主菜单 → 选择 [2-5] 分析模块 → 执行分析 → 显示结果
4. 选择 [6] 生成报告 → 选择保存路径 → 生成 Markdown
5. 选择 [0] 退出
```

### 5.3 子菜单示例（数据清洗）

```
╔══════════════════════════════════════════════════════════╗
║                    数据清洗模块                          ║
╠══════════════════════════════════════════════════════════╣
║  [1] 查看缺失值情况                                      ║
║  [2] 处理缺失值                                          ║
║  [3] 检测异常值                                          ║
║  [4] 删除重复记录                                        ║
║  [5] 数据类型转换                                        ║
║  [0] 返回主菜单                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## 6. 技术选型

| 组件 | 技术 | 用途 |
|------|------|------|
| TUI 框架 | Rich | 表格、进度条、颜色、交互 |
| 数据处理 | pandas, numpy | 数据加载、清洗、分析 |
| 可视化 | matplotlib, seaborn | 图表生成 |
| 机器学习 | scikit-learn | 分类、聚类、回归 |

---

## 7. 依赖清单

```
rich>=13.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
openpyxl>=3.1
```

---

## 8. 输出规范

### 8.1 报告格式 (Markdown)

```markdown
# 数据分析报告

**生成时间**: 2026-05-16 15:30:00
**数据文件**: customer_data.csv
**数据规模**: 10000 行 × 15 列

---

## 一、数据概览

| 指标 | 值 |
|------|-----|
| 总行数 | 10000 |
| 总列数 | 15 |
| 数值列 | 10 |
| 分类列 | 5 |

## 二、统计摘要

（统计表格）

## 三、可视化图表

![折线图](charts/line_chart.png)

## 四、分析结论

（文字总结）
```

### 8.2 输出目录结构

```
output/
├── reports/
│   └── 2026-05-16_15-30-00_report.md
└── charts/
    ├── line_chart.png
    ├── bar_chart.png
    └── heatmap.png
```

---

## 9. 错误处理

| 场景 | 处理方式 |
|------|---------|
| 文件不存在 | 提示用户重新输入 |
| 文件格式不支持 | 提示支持的格式列表 |
| 数据为空 | 提示数据为空，不执行分析 |
| 分析出错 | 捕获异常，显示错误信息，继续执行 |
| 用户取消 | 返回上级菜单 |

---

## 10. 后续扩展

- [ ] 添加更多图表类型（雷达图、地理图等）
- [ ] 支持时间序列数据分析
- [ ] 添加数据导出功能（CSV、Excel）
- [ ] 支持自定义分析模板
- [ ] 添加中英文双语报告
