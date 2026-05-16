"""
可视化模块 - 生成各类统计图表
"""
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _init_charts_dir() -> Path:
    """初始化charts目录，返回目录路径"""
    output_dir = Path(__file__).parent.parent / "output" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_chinese_font():
    """配置中文字体支持"""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STHeiti", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False


def _save_figure(fig: plt.Figure, filename: str, charts_dir: Path) -> str:
    """
    保存图表到指定目录
    
    Args:
        fig: matplotlib图表对象
        filename: 文件名
        charts_dir: 图表保存目录
        
    Returns:
        保存的文件路径
    """
    filepath = charts_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(filepath.absolute())


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, filename: Optional[str] = None) -> str:
    """
    绘制折线图
    
    Args:
        df: 数据框
        x: x轴列名
        y: y轴列名
        title: 图表标题
        filename: 保存文件名，默认为 line_{y}_by_{x}.png
        
    Returns:
        保存路径
    """
    _configure_chinese_font()
    charts_dir = _init_charts_dir()
    
    if filename is None:
        filename = f"line_{y}_by_{x}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, marker="o", ax=ax)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return _save_figure(fig, filename, charts_dir)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, filename: Optional[str] = None) -> str:
    """
    绘制柱状图
    
    Args:
        df: 数据框
        x: x轴列名
        y: y轴列名
        title: 图表标题
        filename: 保存文件名，默认为 bar_{y}_by_{x}.png
        
    Returns:
        保存路径
    """
    _configure_chinese_font()
    charts_dir = _init_charts_dir()
    
    if filename is None:
        filename = f"bar_{y}_by_{x}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    
    return _save_figure(fig, filename, charts_dir)


def plot_pie(df: pd.DataFrame, column: str, title: str, filename: Optional[str] = None) -> str:
    """
    绘制饼图
    
    Args:
        df: 数据框
        column: 用于分组的列名
        title: 图表标题
        filename: 保存文件名，默认为 pie_{column}.png
        
    Returns:
        保存路径
    """
    _configure_chinese_font()
    charts_dir = _init_charts_dir()
    
    if filename is None:
        filename = f"pie_{column}.png"
    
    value_counts = df[column].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(value_counts))
    wedges, texts, autotexts = ax.pie(
        value_counts.values,
        labels=value_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.85
    )
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title(title, fontsize=14, pad=15)
    
    return _save_figure(fig, filename, charts_dir)


def plot_heatmap(df: pd.DataFrame, title: str, filename: Optional[str] = None) -> str:
    """
    绘制热力图
    
    Args:
        df: 数据框（用于计算相关性）
        title: 图表标题
        filename: 保存文件名，默认为 heatmap.png
        
    Returns:
        保存路径
    """
    _configure_chinese_font()
    charts_dir = _init_charts_dir()
    
    if filename is None:
        filename = "heatmap.png"
    
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title(title, fontsize=14, pad=15)
    
    return _save_figure(fig, filename, charts_dir)


def plot_box(df: pd.DataFrame, column: str, title: str, filename: Optional[str] = None) -> str:
    """
    绘制箱线图
    
    Args:
        df: 数据框
        column: 要分析的列名
        title: 图表标题
        filename: 保存文件名，默认为 box_{column}.png
        
    Returns:
        保存路径
    """
    _configure_chinese_font()
    charts_dir = _init_charts_dir()
    
    if filename is None:
        filename = f"box_{column}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=df[column], ax=ax)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel(column, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    return _save_figure(fig, filename, charts_dir)


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: Optional[str] = None) -> str:
    """
    绘制散点图
    
    Args:
        df: 数据框
        x: x轴列名
        y: y轴列名
        title: 图表标题
        filename: 保存文件名，默认为 scatter_{x}_vs_{y}.png
        
    Returns:
        保存路径
    """
    _configure_chinese_font()
    charts_dir = _init_charts_dir()
    
    if filename is None:
        filename = f"scatter_{x}_vs_{y}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, ax=ax, alpha=0.6)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return _save_figure(fig, filename, charts_dir)


def plot_histogram(df: pd.DataFrame, column: str, title: str, bins: int = 30, filename: Optional[str] = None) -> str:
    """
    绘制直方图
    
    Args:
        df: 数据框
        column: 要分析的列名
        bins: 直方图柱数，默认为30
        title: 图表标题
        filename: 保存文件名，默认为 histogram_{column}.png
        
    Returns:
        保存路径
    """
    _configure_chinese_font()
    charts_dir = _init_charts_dir()
    
    if filename is None:
        filename = f"histogram_{column}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=bins, kde=True, ax=ax)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("频数", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    return _save_figure(fig, filename, charts_dir)
