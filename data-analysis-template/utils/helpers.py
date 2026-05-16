"""数据分析和可视化项目的工具函数模块"""

from pathlib import Path
from pandas import DataFrame


def get_output_dir() -> tuple[Path, Path]:
    """
    获取输出目录的路径。
    
    创建并返回 reports 和 charts 两个输出目录的 Path 对象。
    返回的目录路径基于当前文件的父目录向上两级（即项目根目录）。
    
    Returns:
        tuple[Path, Path]: 包含 reports_dir 和 charts_dir 的元组
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    reports_dir = base_dir / "reports"
    charts_dir = base_dir / "charts"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    return reports_dir, charts_dir


def format_bytes(size: int) -> str:
    """
    将字节大小格式化为人类可读字符串。
    
    Args:
        size: 字节大小（整数）
    
    Returns:
        str: 格式化后的大小字符串，如 "1.5 MB"
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def get_numeric_columns(df: DataFrame) -> list:
    """
    获取 DataFrame 中的数值类型列。
    
    Args:
        df: pandas DataFrame 对象
    
    Returns:
        list: 数值列的名称列表
    """
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: DataFrame) -> list:
    """
    获取 DataFrame 中的分类类型列（包含 object、category 和 string 类型）。
    
    Args:
        df: pandas DataFrame 对象
    
    Returns:
        list: 分类列的名称列表
    """
    return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
