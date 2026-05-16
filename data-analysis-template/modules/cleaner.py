"""
数据清洗模块
提供缺失值处理、异常值检测、去重和数据类型转换等功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计DataFrame中各列的缺失值信息
    
    Args:
        df: 目标DataFrame
        
    Returns:
        包含缺失值统计信息的DataFrame，包含以下列：
        - 列名 (Column)
        - 缺失值数量 (Missing Count)
        - 缺失比例 (Missing %)
        - 数据类型 (Data Type)
    """
    console.print(Panel.fit("[bold blue]缺失值统计信息[/bold blue]", border_style="blue"))
    
    total_rows = len(df)
    missing_info = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
        missing_info.append({
            'Column': col,
            'Missing Count': missing_count,
            'Missing %': round(missing_pct, 2),
            'Data Type': str(df[col].dtype)
        })
    
    missing_df = pd.DataFrame(missing_info)
    
    if not missing_df.empty:
        table = Table(title="缺失值统计", show_header=True, header_style="bold magenta")
        table.add_column("列名", style="cyan")
        table.add_column("缺失数量", justify="right", style="yellow")
        table.add_column("缺失比例(%)", justify="right", style="yellow")
        table.add_column("数据类型", style="green")
        
        for _, row in missing_df.iterrows():
            table.add_row(
                str(row['Column']),
                str(row['Missing Count']),
                f"{row['Missing %']:.2f}",
                str(row['Data Type'])
            )
        
        console.print(table)
        console.print(f"\n[bold]总行数:[/bold] {total_rows}")
        console.print(f"[bold]总缺失数:[/bold] {missing_df['Missing Count'].sum()}")
    
    return missing_df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    处理DataFrame中的缺失值
    
    Args:
        df: 目标DataFrame
        strategy: 填充策略，支持以下选项：
            - 'mean': 用均值填充（仅适用于数值列）
            - 'median': 用中位数填充（仅适用于数值列）
            - 'mode': 用众数填充
            - 'forward_fill': 用前向填充
            - 'backward_fill': 用后向填充
            - 'drop': 删除包含缺失值的行
        columns: 需要处理的列名列表，None表示处理所有列
        
    Returns:
        处理后的DataFrame
    """
    console.print(Panel.fit(f"[bold green]缺失值处理 - 策略: {strategy}[/bold green]", border_style="green"))
    
    df_copy = df.copy()
    target_columns = columns if columns else df_copy.columns.tolist()
    
    with console.status("[bold yellow]正在处理缺失值...") as status:
        for col in target_columns:
            if col not in df_copy.columns:
                console.print(f"[yellow]警告: 列 '{col}' 不存在于DataFrame中，跳过[/yellow]")
                continue
            
            missing_count = df_copy[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    fill_value = df_copy[col].mean()
                    df_copy[col].fillna(fill_value, inplace=True)
                    console.print(f"  {col}: 使用均值 {fill_value:.2f} 填充")
                else:
                    console.print(f"  [yellow]警告: 列 '{col}' 不是数值类型，无法使用mean策略[/yellow]")
            
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    fill_value = df_copy[col].median()
                    df_copy[col].fillna(fill_value, inplace=True)
                    console.print(f"  {col}: 使用中位数 {fill_value:.2f} 填充")
                else:
                    console.print(f"  [yellow]警告: 列 '{col}' 不是数值类型，无法使用median策略[/yellow]")
            
            elif strategy == 'mode':
                mode_value = df_copy[col].mode()
                if not mode_value.empty:
                    df_copy[col].fillna(mode_value[0], inplace=True)
                    console.print(f"  {col}: 使用众数 '{mode_value[0]}' 填充")
            
            elif strategy == 'forward_fill':
                df_copy[col].fillna(method='ffill', inplace=True)
                console.print(f"  {col}: 使用前向填充")
            
            elif strategy == 'backward_fill':
                df_copy[col].fillna(method='bfill', inplace=True)
                console.print(f"  {col}: 使用后向填充")
            
            elif strategy == 'drop':
                df_copy.dropna(subset=[col], inplace=True)
                console.print(f"  {col}: 删除包含缺失值的行")
    
    total_missing_after = df_copy.isnull().sum().sum()
    console.print(f"\n[bold]处理后缺失值总数:[/bold] {total_missing_after}")
    
    return df_copy


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用IQR（四分位距）方法检测异常值
    
    Args:
        df: 目标DataFrame
        column: 需要检测的列名
        
    Returns:
        Tuple包含:
        - 异常值DataFrame（包含异常值的数据行）
        - 统计信息DataFrame（包含Q1, Q3, IQR, 下界, 上界等信息）
    """
    console.print(Panel.fit(f"[bold yellow]IQR异常值检测 - 列: {column}[/bold yellow]", border_style="yellow"))
    
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"列 '{column}' 不是数值类型，无法进行异常值检测")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outliers_df = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
    
    stats_df = pd.DataFrame({
        'Statistic': ['Q1 (25%)', 'Q3 (75%)', 'IQR', 'Lower Bound', 'Upper Bound', 
                      'Min', 'Max', 'Outliers Count'],
        'Value': [Q1, Q3, IQR, lower_bound, upper_bound, 
                  df[column].min(), df[column].max(), len(outliers)]
    })
    
    table = Table(title="IQR统计信息", show_header=False, border_style="yellow")
    table.add_column("统计量", style="cyan")
    table.add_column("值", justify="right", style="yellow")
    
    for _, row in stats_df.iterrows():
        if row['Statistic'] == 'Outliers Count':
            table.add_row(row['Statistic'], str(row['Value']), style="bold red")
        else:
            table.add_row(row['Statistic'], f"{row['Value']:.2f}")
    
    console.print(table)
    console.print(f"\n[bold red]检测到 {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.2f}%)[/bold red]")
    
    if len(outliers) > 0:
        console.print("\n[bold red]异常值详情:[/bold red]")
        console.print(outliers[[column]].to_string())
    
    return outliers_df, stats_df


def detect_outliers_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用Z-Score方法检测异常值
    
    Args:
        df: 目标DataFrame
        column: 需要检测的列名
        threshold: Z-Score阈值，默认为3，表示偏离均值超过3个标准差的点为异常值
        
    Returns:
        Tuple包含:
        - 异常值DataFrame（包含异常值的数据行）
        - 统计信息DataFrame（包含均值、标准差等信息）
    """
    console.print(Panel.fit(f"[bold purple]Z-Score异常值检测 - 列: {column} (阈值: {threshold})[/bold purple]", 
                            border_style="purple"))
    
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"列 '{column}' 不是数值类型，无法进行异常值检测")
    
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    
    outliers_df = df[z_scores > threshold].copy()
    outliers_df['_z_score'] = z_scores[z_scores > threshold]
    
    stats_df = pd.DataFrame({
        'Statistic': ['Mean', 'Std', 'Threshold', 'Z-Score Lower Bound', 'Z-Score Upper Bound',
                      'Outliers Count'],
        'Value': [mean, std, threshold, mean - threshold * std, mean + threshold * std, len(outliers_df)]
    })
    
    table = Table(title="Z-Score统计信息", show_header=False, border_style="purple")
    table.add_column("统计量", style="cyan")
    table.add_column("值", justify="right", style="yellow")
    
    for _, row in stats_df.iterrows():
        if row['Statistic'] == 'Outliers Count':
            table.add_row(row['Statistic'], str(row['Value']), style="bold red")
        else:
            table.add_row(row['Statistic'], f"{row['Value']:.4f}")
    
    console.print(table)
    console.print(f"\n[bold red]检测到 {len(outliers_df)} 个异常值 ({len(outliers_df)/len(df)*100:.2f}%)[/bold red]")
    
    if len(outliers_df) > 0:
        console.print("\n[bold red]异常值详情:[/bold red]")
        display_cols = [column, '_z_score']
        console.print(outliers_df[display_cols].to_string())
    
    return outliers_df, stats_df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除DataFrame中的重复行
    
    Args:
        df: 目标DataFrame
        
    Returns:
        移除重复行后的DataFrame
    """
    console.print(Panel.fit("[bold cyan]去重处理[/bold cyan]", border_style="cyan"))
    
    initial_rows = len(df)
    df_copy = df.copy()
    df_copy.drop_duplicates(inplace=True)
    final_rows = len(df_copy)
    removed_rows = initial_rows - final_rows
    
    table = Table(title="去重结果", show_header=False, border_style="cyan")
    table.add_column("项目", style="cyan")
    table.add_column("数量", justify="right", style="yellow")
    
    table.add_row("处理前行数", str(initial_rows))
    table.add_row("处理后行数", str(final_rows))
    table.add_row("删除行数", str(removed_rows), style="bold red")
    
    console.print(table)
    
    if removed_rows > 0:
        console.print(f"[bold green]成功移除 {removed_rows} 个重复行[/bold green]")
    else:
        console.print("[bold]没有发现重复行[/bold]")
    
    return df_copy


def convert_dtype(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
    """
    转换指定列的数据类型
    
    Args:
        df: 目标DataFrame
        column: 需要转换类型的列名
        dtype: 目标数据类型，支持：
            - 'int', 'int64', 'int32': 整数类型
            - 'float', 'float64', 'float32': 浮点数类型
            - 'str', 'string': 字符串类型
            - 'bool', 'boolean': 布尔类型
            - 'datetime': 日期时间类型
            - 'category': 分类类型
        
    Returns:
        转换类型后的DataFrame
    """
    console.print(Panel.fit(f"[bold magenta]数据类型转换 - 列: {column} -> {dtype}[/bold magenta]", 
                            border_style="magenta"))
    
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    df_copy = df.copy()
    original_dtype = str(df_copy[column].dtype)
    
    try:
        with console.status(f"[bold yellow]正在转换 {column} 的数据类型...") as status:
            df_copy[column] = df_copy[column].astype(dtype)
        
        table = Table(title="类型转换结果", show_header=False, border_style="magenta")
        table.add_column("项目", style="cyan")
        table.add_column("值", style="yellow")
        
        table.add_row("列名", column)
        table.add_row("原始类型", original_dtype)
        table.add_row("目标类型", str(dtype))
        table.add_row("转换状态", "[bold green]成功[/bold green]")
        
        console.print(table)
        
        console.print(f"\n[bold]转换前后数据对比:[/bold]")
        console.print(f"原始类型: {original_dtype}")
        console.print(f"新类型: {str(df_copy[column].dtype)}")
        console.print(f"\n[bold]数据示例:[/bold]")
        console.print(df_copy[column].head(10).to_string())
        
    except Exception as e:
        console.print(f"[bold red]转换失败: {str(e)}[/bold red]")
        raise
    
    return df_copy
