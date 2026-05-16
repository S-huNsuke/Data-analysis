"""
数据加载模块
提供 CSV 和 Excel 文件的加载功能
"""

import os
from typing import Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def load_data(file_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    加载数据文件（CSV 或 Excel）

    Args:
        file_path: 文件路径

    Returns:
        Tuple[DataFrame, dict]: DataFrame 数据和基本信息字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in [".csv", ".xlsx", ".xls"]:
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .csv, .xlsx, .xls")

    df = None
    encodings = ["utf-8", "gbk", "gb2312", "gb18030"]

    if file_ext == ".csv":
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                console.print(f"[green]✓[/green] 成功加载 CSV 文件（编码: {encoding}）")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise RuntimeError(f"读取 CSV 文件失败: {e}")

        if df is None:
            raise RuntimeError("无法读取 CSV 文件，请检查文件编码")

    elif file_ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(file_path, engine="openpyxl" if file_ext == ".xlsx" else "xlrd")
            console.print(f"[green]✓[/green] 成功加载 Excel 文件")
        except Exception as e:
            raise RuntimeError(f"读取 Excel 文件失败: {e}")

    info = get_data_info(df)
    console.print(f"[cyan]📊[/cyan] 数据维度: {info['shape'][0]} 行 × {info['shape'][1]} 列")

    return df, info


def get_data_info(df: pd.DataFrame) -> dict:
    """
    获取 DataFrame 的基本信息

    Args:
        df: pandas DataFrame 对象

    Returns:
        dict: 包含基本信息字段的字典
              - shape: 数据维度
              - columns: 列名列表
              - dtypes: 列数据类型字典
              - missing: 缺失值统计
              - numeric_columns: 数值列列表
              - categorical_columns: 类别列列表
              - memory_usage: 内存占用
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "memory_usage": df.memory_usage(deep=True).sum()
    }
    return info


def preview_data(df: pd.DataFrame, n: int = 5) -> None:
    """
    预览 DataFrame 数据，使用 Rich 表格展示

    Args:
        df: pandas DataFrame 对象
        n: 预览行数，默认 5 行
    """
    console.print("\n[bold cyan]数据预览（前 {} 行）[/bold cyan]".format(n))

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("列名", style="cyan", no_wrap=True)

    for col in df.columns:
        table.add_column(str(col), style="white")

    display_df = df.head(n)
    for idx, row in display_df.iterrows():
        table.add_row(str(idx), *[str(val) for val in row.values])

    console.print(table)

    console.print(f"\n[dim]共 {len(df)} 行，当前显示前 {min(n, len(df))} 行[/dim]")
