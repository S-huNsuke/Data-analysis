"""探索性数据分析模块（Exploratory Data Analysis）

提供数据探索的基本分析功能，包括描述性统计、相关性分析、
分组统计和分布分析。
"""

import pandas as pd
import numpy as np
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


console = Console()


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算描述性统计信息
    
    对数值型列计算基本的描述性统计指标。
    
    Args:
        df: 输入的 pandas DataFrame
        
    Returns:
        包含描述性统计的 DataFrame，包含以下列：
        - count: 非空值数量
        - mean: 平均值
        - std: 标准差
        - min: 最小值
        - 25%: 25%分位数
        - 50%: 中位数
        - 75%: 75%分位数
        - max: 最大值
    """
    console.print(Panel("[bold cyan]描述性统计[/bold cyan]", expand=False))
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        console.print("[yellow]警告：数据框中没有数值型列[/yellow]")
        return pd.DataFrame()
    
    stats_df = numeric_df.describe().T
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("列名", style="cyan")
    table.add_column("计数", justify="right", style="green")
    table.add_column("均值", justify="right", style="yellow")
    table.add_column("标准差", justify="right", style="blue")
    table.add_column("最小值", justify="right", style="red")
    table.add_column("中位数", justify="right", style="purple")
    table.add_column("最大值", justify="right", style="green")
    
    for col in stats_df.index:
        row = stats_df.loc[col]
        table.add_row(
            str(col),
            f"{row['count']:.0f}",
            f"{row['mean']:.2f}",
            f"{row['std']:.2f}",
            f"{row['min']:.2f}",
            f"{row['50%']:.2f}",
            f"{row['max']:.2f}"
        )
    
    console.print(table)
    
    return stats_df


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """计算相关性矩阵
    
    计算数值型列之间的皮尔逊相关系数矩阵。
    
    Args:
        df: 输入的 pandas DataFrame
        
    Returns:
        相关系数矩阵 DataFrame
    """
    console.print(Panel("[bold cyan]相关性分析[/bold cyan]", expand=False))
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        console.print("[yellow]警告：数据框中没有数值型列[/yellow]")
        return pd.DataFrame()
    
    corr_matrix = numeric_df.corr()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("变量", style="cyan", no_wrap=True)
    
    for col in corr_matrix.columns:
        table.add_column(col[:10], justify="right", style="yellow")
    
    for idx, row_name in enumerate(corr_matrix.index):
        row_values = []
        for val in corr_matrix.iloc[idx]:
            if pd.isna(val):
                row_values.append("N/A")
            else:
                color = "green" if val > 0 else "red"
                row_values.append(f"[{color}]{val:.2f}[/{color}]")
        
        table.add_row(str(row_name)[:15], *row_values)
    
    console.print(table)
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        console.print("\n[bold yellow]高相关性变量对（|r| > 0.7）：[/bold yellow]")
        for var1, var2, corr in high_corr_pairs:
            console.print(f"  • {var1} <-> {var2}: {corr:.3f}")
    
    return corr_matrix


def group_analysis(
    df: pd.DataFrame,
    group_col: str,
    agg_col: str,
    agg_func: str = 'mean'
) -> pd.DataFrame:
    """执行分组统计分析
    
    根据指定的分组列和聚合函数计算分组统计。
    
    Args:
        df: 输入的 pandas DataFrame
        group_col: 分组依据的列名
        agg_col: 需要聚合的列名
        agg_func: 聚合函数，支持 'mean', 'sum', 'count', 
                  'min', 'max', 'std', 'median'
        
    Returns:
        分组统计结果的 DataFrame
    """
    console.print(Panel(f"[bold cyan]分组统计[/bold cyan] - 按 '{group_col}' 分组", expand=False))
    
    if group_col not in df.columns:
        console.print(f"[red]错误：列 '{group_col}' 不存在[/red]")
        return pd.DataFrame()
    
    if agg_col not in df.columns:
        console.print(f"[red]错误：列 '{agg_col}' 不存在[/red]")
        return pd.DataFrame()
    
    agg_funcs = {
        'mean': 'mean',
        'sum': 'sum',
        'count': 'count',
        'min': 'min',
        'max': 'max',
        'std': 'std',
        'median': 'median'
    }
    
    if agg_func not in agg_funcs:
        console.print(f"[yellow]警告：不支持的聚合函数 '{agg_func}'，使用 'mean'[/yellow]")
        agg_func = 'mean'
    
    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    result.columns = [group_col, f'{agg_col}_{agg_func}']
    result = result.sort_values(f'{agg_col}_{agg_func}', ascending=False)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column(group_col, style="cyan")
    table.add_column(f"{agg_col}_{agg_func}", justify="right", style="green")
    
    for _, row in result.head(20).iterrows():
        table.add_row(
            str(row[group_col])[:30],
            f"{row[f'{agg_col}_{agg_func}']:.2f}"
        )
    
    console.print(table)
    
    if len(result) > 20:
        console.print(f"[dim]（显示前 20 行，共 {len(result)} 行）[/dim]")
    
    return result


def distribution_analysis(df: pd.DataFrame, column: str) -> dict:
    """分析数据分布特征
    
    计算并展示指定列的分布统计信息，包括偏度和峰度。
    
    Args:
        df: 输入的 pandas DataFrame
        column: 需要分析的列名
        
    Returns:
        包含分布统计信息的字典：
        - count: 样本数量
        - mean: 平均值
        - median: 中位数
        - std: 标准差
        - skewness: 偏度
        - kurtosis: 峰度
        - min: 最小值
        - max: 最大值
        - q1: 25%分位数
        - q3: 75%分位数
        - iqr: 四分位距
        - missing: 缺失值数量
    """
    console.print(Panel(f"[bold cyan]分布分析[/bold cyan] - 列: '{column}'", expand=False))
    
    if column not in df.columns:
        console.print(f"[red]错误：列 '{column}' 不存在[/red]")
        return {}
    
    col_data = df[column]
    
    if col_data.dtype not in [np.number, 'int64', 'float64']:
        console.print(f"[yellow]警告：列 '{column}' 不是数值型[/yellow]")
        return {}
    
    count = col_data.count()
    mean = col_data.mean()
    median = col_data.median()
    std = col_data.std()
    skewness = col_data.skew()
    kurtosis = col_data.kurtosis()
    min_val = col_data.min()
    max_val = col_data.max()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    missing = col_data.isna().sum()
    
    info_table = Table(show_header=False, box=None)
    info_table.add_column("指标", style="cyan")
    info_table.add_column("值", justify="right", style="yellow")
    
    info_table.add_row("样本数量", f"{count:.0f}")
    info_table.add_row("平均值", f"{mean:.2f}")
    info_table.add_row("中位数", f"{median:.2f}")
    info_table.add_row("标准差", f"{std:.2f}")
    info_table.add_row("最小值", f"{min_val:.2f}")
    info_table.add_row("最大值", f"{max_val:.2f}")
    info_table.add_row("25%分位数", f"{q1:.2f}")
    info_table.add_row("75%分位数", f"{q3:.2f}")
    info_table.add_row("四分位距", f"{iqr:.2f}")
    info_table.add_row("偏度", f"{skewness:.3f}")
    info_table.add_row("峰度", f"{kurtosis:.3f}")
    info_table.add_row("缺失值", f"{missing}")
    
    console.print(info_table)
    
    skewness_note = ""
    if skewness > 1:
        skewness_note = "[yellow]右偏分布[/yellow]"
    elif skewness < -1:
        skewness_note = "[yellow]左偏分布[/yellow]"
    else:
        skewness_note = "[green]近似对称分布[/green]"
    
    console.print(f"\n偏度解读: {skewness_note}")
    
    return {
        'count': count,
        'mean': mean,
        'median': median,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': min_val,
        'max': max_val,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'missing': missing
    }
