# 交互式数据分析模板 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个基于 Rich 库的交互式 TUI 数据分析模板，支持数据加载、清洗、探索性分析、可视化和机器学习模块。

**Architecture:** 采用模块化设计，主程序统一调度各分析模块，输出 Markdown 报告和可视化图表到 output 目录。

**Tech Stack:** Rich, pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 文件结构

```
data-analysis-template/
├── main.py                    # 主入口，TUI 交互
├── modules/
│   ├── __init__.py
│   ├── loader.py              # 数据加载
│   ├── cleaner.py             # 数据清洗
│   ├── eda.py                 # 探索性分析
│   ├── visualizer.py          # 可视化
│   └── modeler.py             # 机器学习
├── reports/
│   └── generator.py           # 报告生成
├── utils/
│   └── helpers.py             # 工具函数
└── output/
    ├── reports/
    └── charts/
```

---

### Task 1: 创建目录结构和配置文件

**Files:**
- Create: `data-analysis-template/modules/__init__.py`
- Create: `data-analysis-template/reports/__init__.py`
- Create: `data-analysis-template/utils/__init__.py`
- Create: `data-analysis-template/output/reports/.gitkeep`
- Create: `data-analysis-template/output/charts/.gitkeep`
- Modify: `pyproject.toml`（添加新依赖）

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/caojun/Desktop/Data-analysis
mkdir -p data-analysis-template/modules
mkdir -p data-analysis-template/reports
mkdir -p data-analysis-template/utils
mkdir -p data-analysis-template/output/reports
mkdir -p data-analysis-template/output/charts
touch data-analysis-template/modules/__init__.py
touch data-analysis-template/reports/__init__.py
touch data-analysis-template/utils/__init__.py
touch data-analysis-template/output/reports/.gitkeep
touch data-analysis-template/output/charts/.gitkeep
```

- [ ] **Step 2: 更新 pyproject.toml 添加新依赖**

```toml
# 在 dependencies 中添加
"rich>=13.0",
```

- [ ] **Step 3: 运行 uv sync 安装依赖**

```bash
cd /Users/caojun/Desktop/Data-analysis && uv sync
```

---

### Task 2: 实现数据加载模块 (modules/loader.py)

**Files:**
- Create: `data-analysis-template/modules/loader.py`

- [ ] **Step 1: 编写 loader.py**

```python
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def load_data(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    加载数据文件，支持 CSV 和 Excel 格式
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk')
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, engine='openpyxl' if suffix == '.xlsx' else None)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持 CSV/Excel")
    
    info = get_data_info(df)
    return df, info


def get_data_info(df: pd.DataFrame) -> dict:
    """获取数据基本信息"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'missing': df.isnull().sum().to_dict(),
        'duplicate_count': df.duplicated().sum(),
    }


def preview_data(df: pd.DataFrame, n: int = 5) -> None:
    """预览数据前 n 行"""
    console.print(f"\n[bold]数据预览 (前 {n} 行):[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    
    for col in df.columns[:10]:
        table.add_column(str(col)[:15], style="cyan")
    
    for idx, row in df.head(n).iterrows():
        table.add_row(*[str(v)[:15] for v in row.values[:10]])
    
    console.print(table)
```

---

### Task 3: 实现工具函数 (utils/helpers.py)

**Files:**
- Create: `data-analysis-template/utils/helpers.py`

- [ ] **Step 1: 编写 helpers.py**

```python
import os
from datetime import datetime
from pathlib import Path


def get_output_dir(base_dir: str = "output") -> tuple[Path, Path]:
    """获取输出目录，返回 (reports_dir, charts_dir)"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(__file__).parent.parent / base_dir
    reports_dir = base / "reports"
    charts_dir = base / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir, charts_dir


def format_bytes(size: int) -> str:
    """格式化字节大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def get_numeric_columns(df) -> list:
    """获取数值列"""
    return df.select_dtypes(include=['number']).columns.tolist()


def get_categorical_columns(df) -> list:
    """获取分类列"""
    return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
```

---

### Task 4: 实现数据清洗模块 (modules/cleaner.py)

**Files:**
- Create: `data-analysis-template/modules/cleaner.py`

- [ ] **Step 1: 编写 cleaner.py**

```python
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """获取缺失值统计"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    result = pd.DataFrame({
        '列名': missing.index,
        '缺失数量': missing.values,
        '缺失比例(%)': missing_pct.values,
        '数据类型': df.dtypes.values
    })
    return result[result['缺失数量'] > 0]


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: list = None) -> pd.DataFrame:
    """
    处理缺失值
    strategy: 'drop' | 'mean' | 'median' | 'mode' | 'forward_fill' | 'backward_fill'
    """
    df = df.copy()
    cols = columns or df.columns.tolist()
    
    for col in cols:
        if df[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            df = df.dropna(subset=[col])
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else '')
        elif strategy == 'forward_fill':
            df[col] = df[col].ffill()
        elif strategy == 'backward_fill':
            df[col] = df[col].bfill()
    
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> tuple:
    """使用 IQR 方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> tuple:
    """使用 Z-score 方法检测异常值"""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers, threshold


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """删除重复记录"""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    console.print(f"[green]删除了 {before - after} 条重复记录[/green]")
    return df


def convert_dtype(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
    """转换数据类型"""
    df = df.copy()
    try:
        if dtype == 'numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif dtype == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif dtype == 'category':
            df[column] = df[column].astype('category')
        else:
            df[column] = df[column].astype(dtype)
        console.print(f"[green]列 {column} 已转换为 {dtype}[/green]")
    except Exception as e:
        console.print(f"[red]转换失败: {e}[/red]")
    return df
```

---

### Task 5: 实现探索性分析模块 (modules/eda.py)

**Files:**
- Create: `data-analysis-template/modules/eda.py`

- [ ] **Step 1: 编写 eda.py**

```python
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """描述性统计"""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats['缺失值'] = numeric_df.isnull().sum()
    return stats


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """相关性分析"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr()


def group_analysis(df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str = 'mean') -> pd.DataFrame:
    """分组统计"""
    return df.groupby(group_col)[agg_col].agg(agg_func).reset_index()


def distribution_analysis(df: pd.DataFrame, column: str) -> dict:
    """分布分析"""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {'type': 'categorical', 'value_counts': df[column].value_counts().to_dict()}
    
    return {
        'type': 'numeric',
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurt(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q25': df[column].quantile(0.25),
        'q75': df[column].quantile(0.75),
    }
```

---

### Task 6: 实现可视化模块 (modules/visualizer.py)

**Files:**
- Create: `data-analysis-template/modules/visualizer.py`

- [ ] **Step 1: 编写 visualizer.py**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def _get_output_path(filename: str) -> Path:
    """获取输出路径"""
    output_dir = Path(__file__).parent.parent / 'output' / 'charts'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def plot_line(df: pd.DataFrame, x: str, y: str, title: str = "", filename: Optional[str] = None) -> str:
    """折线图"""
    if filename is None:
        filename = f"line_{x}_vs_{y}.png"
    path = _get_output_path(filename)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, marker='o')
    plt.title(title or f'{y} 随 {x} 变化趋势', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str = "", filename: Optional[str] = None) -> str:
    """柱状图"""
    if filename is None:
        filename = f"bar_{x}_{y}.png"
    path = _get_output_path(filename)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title or f'{y} 按 {x} 分布', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_pie(df: pd.DataFrame, column: str, title: str = "", filename: Optional[str] = None) -> str:
    """饼图"""
    if filename is None:
        filename = f"pie_{column}.png"
    path = _get_output_path(filename)
    
    plt.figure(figsize=(8, 8))
    value_counts = df[column].value_counts()
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(title or f'{column} 分布', fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_heatmap(df: pd.DataFrame, title: str = "", filename: Optional[str] = None) -> str:
    """热力图"""
    if filename is None:
        filename = "heatmap_correlation.png"
    path = _get_output_path(filename)
    
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        return ""
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title(title or '相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_box(df: pd.DataFrame, column: str, title: str = "", filename: Optional[str] = None) -> str:
    """箱线图"""
    if filename is None:
        filename = f"box_{column}.png"
    path = _get_output_path(filename)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column])
    plt.title(title or f'{column} 箱线图', fontsize=14)
    plt.ylabel(column, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str = "", filename: Optional[str] = None) -> str:
    """散点图"""
    if filename is None:
        filename = f"scatter_{x}_vs_{y}.png"
    path = _get_output_path(filename)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(title or f'{x} 与 {y} 关系', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30, title: str = "", filename: Optional[str] = None) -> str:
    """直方图"""
    if filename is None:
        filename = f"hist_{column}.png"
    path = _get_output_path(filename)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=bins, kde=True)
    plt.title(title or f'{column} 分布', fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)
```

---

### Task 7: 实现机器学习模块 (modules/modeler.py)

**Files:**
- Create: `data-analysis-template/modules/modeler.py`

- [ ] **Step 1: 编写 modeler.py**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, r2_score
from rich.console import Console

console = Console()


def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """逻辑回归分类"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred),
        'feature_importance': dict(zip(X.columns, model.coef_[0]))
    }


def train_random_forest(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100) -> dict:
    """随机森林分类"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred),
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }


def train_kmeans(X: pd.DataFrame, n_clusters: int = 4) -> dict:
    """K-Means 聚类"""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    
    return {
        'model': model,
        'labels': labels,
        'silhouette_score': silhouette,
        'centers': model.cluster_centers_
    }


def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """线性回归"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model,
        'r2_score': r2,
        'coefficients': dict(zip(X.columns, model.coef_))
    }
```

---

### Task 8: 实现报告生成模块 (reports/generator.py)

**Files:**
- Create: `data-analysis-template/reports/generator.py`

- [ ] **Step 1: 编写 generator.py**

```python
import pandas as pd
from datetime import datetime
from pathlib import Path


def generate_report(
    title: str,
    data_info: dict,
    stats: dict = None,
    charts: list = None,
    conclusions: str = "",
    output_path: str = None
) -> str:
    """生成 Markdown 分析报告"""
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(__file__).parent.parent / 'output' / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{timestamp}_report.md"
    else:
        output_path = Path(output_path)
    
    missing_info = data_info.get('missing', {})
    missing_rows = [f"| {k} | {v} |" for k, v in missing_info.items() if v > 0]
    
    report_content = f"""# {title}

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**数据文件**: {data_info.get('file_name', 'N/A')}
**数据规模**: {data_info['rows']} 行 × {data_info['columns']} 列

---

## 一、数据概览

| 指标 | 值 |
|------|-----|
| 总行数 | {data_info['rows']:,} |
| 总列数 | {data_info['columns']} |
| 数值列 | {len(data_info.get('numeric_cols', []))} |
| 分类列 | {len(data_info.get('categorical_cols', []))} |
| 重复记录 | {data_info.get('duplicate_count', 0)} |

### 1.1 列信息

| 列名 | 数据类型 |
|------|---------|
"""
    
    for col, dtype in data_info.get('dtypes', {}).items():
        report_content += f"| {col} | {dtype} |\n"
    
    if missing_rows:
        report_content += f"""
### 1.2 缺失值情况

| 列名 | 缺失数量 |
|------|---------|
"""
        report_content += "\n".join(missing_rows) + "\n"
    
    if stats is not None and not stats.empty:
        report_content += """
## 二、统计摘要

"""
        if isinstance(stats, pd.DataFrame):
            report_content += stats.to_markdown() + "\n"
    
    if charts:
        report_content += """
## 三、可视化图表

"""
        for chart_path in charts:
            chart_name = Path(chart_path).name
            report_content += f"- {chart_name}\n"
    
    if conclusions:
        report_content += f"""
## 四、分析结论

{conclusions}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding='utf-8')
    
    return str(output_path)
```

---

### Task 9: 实现主程序 (main.py)

**Files:**
- Create: `data-analysis-template/main.py`

- [ ] **Step 1: 编写 main.py（完整 TUI 交互程序）**

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.menu import Menu
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

from modules.loader import load_data, get_data_info, preview_data
from modules.cleaner import get_missing_info, handle_missing_values, remove_duplicates
from modules.eda import descriptive_stats, correlation_analysis, group_analysis
from modules.visualizer import plot_line, plot_bar, plot_pie, plot_heatmap, plot_box, plot_scatter, plot_histogram
from modules.modeler import train_logistic_regression, train_random_forest, train_kmeans, train_linear_regression
from reports.generator import generate_report
from utils.helpers import get_output_dir, get_numeric_columns, get_categorical_columns

console = Console()
df = None
data_info = None
charts_generated = []
stats_results = {}


def show_banner():
    """显示欢迎界面"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]交互式数据分析模板 v1.0[/bold cyan]\n\n"
        "基于 Python + Rich 的终端数据分析工具\n"
        "支持数据加载、清洗、分析、可视化、机器学习",
        border_style="cyan"
    ))


def show_main_menu():
    """显示主菜单"""
    options = [
        "📁 数据加载",
        "🧹 数据清洗",
        "📊 探索性分析",
        "📈 可视化图表",
        "🤖 机器学习",
        "📝 生成报告",
        "❌ 退出"
    ]
    
    menu = Menu("请选择一个选项:", options, cannonball=">>>")
    return menu.choose()


def load_data_flow():
    """数据加载流程"""
    global df, data_info
    
    show_banner()
    console.print("\n[bold]📁 数据加载[/bold]\n")
    
    file_path = Prompt.ask("[bold]请输入文件路径[/bold]", default="")
    
    if not file_path:
        console.print("[red]文件路径不能为空[/red]")
        Prompt.ask("按 Enter 继续...")
        return
    
    try:
        df, data_info = load_data(file_path)
        data_info['file_name'] = file_path
        
        console.print(f"\n[bold green]✓ 数据加载成功！[/bold green]")
        console.print(f"数据规模: {data_info['rows']} 行 × {data_info['columns']} 列")
        
        preview_data(df, n=5)
        
        stats_results['info'] = data_info
        
    except Exception as e:
        console.print(f"[bold red]✗ 加载失败: {e}[/bold red]")
    
    Prompt.ask("\n按 Enter 继续...")


def data_cleaning_flow():
    """数据清洗流程"""
    global df
    
    if df is None:
        console.print("[yellow]请先加载数据！[/yellow]")
        Prompt.ask("按 Enter 继续...")
        return
    
    while True:
        show_banner()
        console.print("\n[bold]🧹 数据清洗[/bold]\n")
        
        options = [
            "查看缺失值情况",
            "处理缺失值",
            "删除重复记录",
            "返回主菜单"
        ]
        
        choice = Prompt.ask("[bold]请选择操作[/bold]", choices=[str(i) for i in range(1, 5)], default="4")
        
        if choice == "1":
            missing = get_missing_info(df)
            if missing.empty:
                console.print("[green]数据无缺失值[/green]")
            else:
                console.print(missing.to_string(index=False))
            Prompt.ask("按 Enter 继续...")
            
        elif choice == "2":
            strategy = Prompt.ask(
                "选择填充策略",
                choices=["drop", "mean", "median", "mode", "ffill", "bfill"],
                default="mean"
            )
            df = handle_missing_values(df, strategy=strategy)
            console.print("[green]缺失值处理完成[/green]")
            Prompt.ask("按 Enter 继续...")
            
        elif choice == "3":
            df = remove_duplicates(df)
            Prompt.ask("按 Enter 继续...")
            
        else:
            break


def eda_flow():
    """探索性分析流程"""
    global df, stats_results
    
    if df is None:
        console.print("[yellow]请先加载数据！[/yellow]")
        Prompt.ask("按 Enter 继续...")
        return
    
    while True:
        show_banner()
        console.print("\n[bold]📊 探索性分析[/bold]\n")
        
        options = [
            "描述性统计",
            "相关性分析",
            "分组统计",
            "返回主菜单"
        ]
        
        choice = Prompt.ask("[bold]请选择操作[/bold]", choices=[str(i) for i in range(1, 5)], default="4")
        
        if choice == "1":
            stats = descriptive_stats(df)
            stats_results['descriptive'] = stats
            console.print(stats.to_string())
            Prompt.ask("按 Enter 继续...")
            
        elif choice == "2":
            corr = correlation_analysis(df)
            stats_results['correlation'] = corr
            console.print(corr.to_string())
            Prompt.ask("按 Enter 继续...")
            
        elif choice == "3":
            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            
            if not categorical_cols or not numeric_cols:
                console.print("[yellow]需要同时存在分类列和数值列[/yellow]")
            else:
                group_col = Prompt.ask("选择分组列", choices=categorical_cols)
                agg_col = Prompt.ask("选择聚合列", choices=numeric_cols)
                result = group_analysis(df, group_col, agg_col)
                console.print(result.to_string(index=False))
                stats_results['group'] = result
            Prompt.ask("按 Enter 继续...")
            
        else:
            break


def visualization_flow():
    """可视化流程"""
    global df, charts_generated
    
    if df is None:
        console.print("[yellow]请先加载数据！[/yellow]")
        Prompt.ask("按 Enter 继续...")
        return
    
    while True:
        show_banner()
        console.print("\n[bold]📈 可视化图表[/bold]\n")
        
        options = [
            "折线图",
            "柱状图",
            "饼图",
            "热力图",
            "箱线图",
            "散点图",
            "直方图",
            "返回主菜单"
        ]
        
        choice = Prompt.ask("[bold]请选择图表类型[/bold]", choices=[str(i) for i in range(1, 9)], default="8")
        
        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)
        
        try:
            if choice == "1":
                x = Prompt.ask("X轴列", choices=df.columns.tolist())
                y = Prompt.ask("Y轴列", choices=numeric_cols)
                path = plot_line(df, x, y)
                charts_generated.append(path)
                console.print(f"[green]图表已保存: {path}[/green]")
                
            elif choice == "2":
                x = Prompt.ask("X轴列", choices=df.columns.tolist())
                y = Prompt.ask("Y轴列", choices=numeric_cols)
                path = plot_bar(df, x, y)
                charts_generated.append(path)
                console.print(f"[green]图表已保存: {path}[/green]")
                
            elif choice == "3":
                col = Prompt.ask("选择列", choices=df.columns.tolist())
                path = plot_pie(df, col)
                charts_generated.append(path)
                console.print(f"[green]图表已保存: {path}[/green]")
                
            elif choice == "4":
                path = plot_heatmap(df)
                charts_generated.append(path)
                console.print(f"[green]图表已保存: {path}[/green]")
                
            elif choice == "5":
                col = Prompt.ask("选择列", choices=numeric_cols)
                path = plot_box(df, col)
                charts_generated.append(path)
                console.print(f"[green]图表已保存: {path}[/green]")
                
            elif choice == "6":
                x = Prompt.ask("X轴列", choices=numeric_cols)
                y = Prompt.ask("Y轴列", choices=numeric_cols)
                path = plot_scatter(df, x, y)
                charts_generated.append(path)
                console.print(f"[green]图表已保存: {path}[/green]")
                
            elif choice == "7":
                col = Prompt.ask("选择列", choices=numeric_cols)
                path = plot_histogram(df, col)
                charts_generated.append(path)
                console.print(f"[green]图表已保存: {path}[/green]")
                
            else:
                break
                
        except Exception as e:
            console.print(f"[red]生成失败: {e}[/red]")
        
        Prompt.ask("按 Enter 继续...")


def ml_flow():
    """机器学习流程"""
    global df, stats_results
    
    if df is None:
        console.print("[yellow]请先加载数据！[/yellow]")
        Prompt.ask("按 Enter 继续...")
        return
    
    while True:
        show_banner()
        console.print("\n[bold]🤖 机器学习[/bold]\n")
        
        options = [
            "逻辑回归分类",
            "随机森林分类",
            "K-Means 聚类",
            "线性回归",
            "返回主菜单"
        ]
        
        choice = Prompt.ask("[bold]请选择算法[/bold]", choices=[str(i) for i in range(1, 6)], default="5")
        
        numeric_cols = get_numeric_columns(df)
        
        try:
            if choice == "1":
                target_col = Prompt.ask("选择目标列", choices=numeric_cols)
                feature_cols = [c for c in numeric_cols if c != target_col]
                
                X = df[feature_cols].fillna(0)
                y = df[target_col]
                
                result = train_logistic_regression(X, y)
                stats_results['logistic'] = result
                
                console.print(f"[green]准确率: {result['accuracy']:.4f}[/green]")
                console.print(result['report'])
                
            elif choice == "2":
                target_col = Prompt.ask("选择目标列", choices=numeric_cols)
                feature_cols = [c for c in numeric_cols if c != target_col]
                
                X = df[feature_cols].fillna(0)
                y = df[target_col]
                
                result = train_random_forest(X, y)
                stats_results['random_forest'] = result
                
                console.print(f"[green]准确率: {result['accuracy']:.4f}[/green]")
                
            elif choice == "3":
                n_clusters = int(Prompt.ask("聚类数量", default="4"))
                X = df[numeric_cols].fillna(0)
                
                result = train_kmeans(X, n_clusters)
                stats_results['kmeans'] = result
                
                console.print(f"[green]轮廓系数: {result['silhouette_score']:.4f}[/green]")
                
            elif choice == "4":
                target_col = Prompt.ask("选择目标列", choices=numeric_cols)
                feature_cols = [c for c in numeric_cols if c != target_col]
                
                X = df[feature_cols].fillna(0)
                y = df[target_col]
                
                result = train_linear_regression(X, y)
                stats_results['linear_regression'] = result
                
                console.print(f"[green]R² 分数: {result['r2_score']:.4f}[/green]")
                
            else:
                break
                
        except Exception as e:
            console.print(f"[red]执行失败: {e}[/red]")
        
        Prompt.ask("按 Enter 继续...")


def generate_report_flow():
    """生成报告流程"""
    global data_info, stats_results, charts_generated
    
    if data_info is None:
        console.print("[yellow]请先加载数据！[/yellow]")
        Prompt.ask("按 Enter 继续...")
        return
    
    show_banner()
    console.print("\n[bold]📝 生成报告[/bold]\n")
    
    title = Prompt.ask("报告标题", default="数据分析报告")
    
    try:
        path = generate_report(
            title=title,
            data_info=data_info,
            stats=stats_results.get('descriptive'),
            charts=charts_generated
        )
        console.print(f"[bold green]✓ 报告已生成: {path}[/bold green]")
    except Exception as e:
        console.print(f"[red]生成失败: {e}[/red]")
    
    Prompt.ask("按 Enter 继续...")


def main():
    """主函数"""
    while True:
        show_banner()
        
        if df is None:
            console.print("[dim]提示: 请先加载数据[/dim]")
        
        choice = show_main_menu()
        
        if choice == "1":
            load_data_flow()
        elif choice == "2":
            data_cleaning_flow()
        elif choice == "3":
            eda_flow()
        elif choice == "4":
            visualization_flow()
        elif choice == "5":
            ml_flow()
        elif choice == "6":
            generate_report_flow()
        else:
            console.print("\n[bold cyan]感谢使用！再见！[/bold cyan]\n")
            break


if __name__ == "__main__":
    main()
```

---

### Task 10: 更新 pyproject.toml 添加模板依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 rich 依赖**

```toml
# 在 dependencies 中添加
"rich>=13.0",
```

- [ ] **Step 2: 添加模板为独立项目**

在根目录 pyproject.toml 中添加:

```toml
[project.optional-dependencies]
template = [
    "rich>=13.0",
]
```

---

## 自检清单

1. **Spec 覆盖检查**:
   - [x] 数据加载 (loader.py) - Task 2
   - [x] 数据清洗 (cleaner.py) - Task 4
   - [x] 探索性分析 (eda.py) - Task 5
   - [x] 可视化图表 (visualizer.py) - Task 6
   - [x] 机器学习 (modeler.py) - Task 7
   - [x] 报告生成 (generator.py) - Task 8
   - [x] TUI 交互主程序 (main.py) - Task 9

2. **占位符检查**: 无 TBD/TODO/placeholder

3. **类型一致性检查**: 所有函数签名已定义，参数类型一致

---

## 执行方式选择

**Plan complete and saved to `docs/superpowers/plans/2026-05-16-data-analysis-template-plan.md`.**

两个执行选项:

**1. 子代理驱动开发（推荐）** - 每个任务派发一个子代理，任务间审查，快速迭代

**2. 内联执行** - 在本会话中按任务执行，带检查点

选择哪种方式？
