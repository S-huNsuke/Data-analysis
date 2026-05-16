#!/usr/bin/env python3
"""
数据分析和可视化工具 - 交互式 TUI 主程序

提供完整的数据分析工作流，包括数据加载、清洗、探索性分析、
可视化、机器学习和报告生成等功能。
"""

import sys
import os
import signal
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text

from modules import loader, cleaner, eda, visualizer, modeler
from reports import generator

console = Console()

df: Optional[pd.DataFrame] = None
data_info: Optional[Dict[str, Any]] = None
stats_results: Optional[Dict[str, Any]] = None
charts_generated: list = []
current_file_path: Optional[str] = None


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号，安全退出程序"""
    console.print("\n\n[bold yellow]收到退出信号，正在安全退出...[/bold yellow]")
    console.print("[bold cyan]感谢使用数据分析工具，再见！[/bold cyan]")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def show_welcome():
    """显示欢迎界面和 Banner"""
    console.clear()
    
    banner_text = Text()
    banner_text.append("╔══════════════════════════════════════════════════════════════╗\n", style="bold cyan")
    banner_text.append("║                                                              ║\n", style="bold cyan")
    banner_text.append("║           📊 数据分析与可视化工具 (Data Analysis Tool)        ║\n", style="bold magenta")
    banner_text.append("║                                                              ║\n", style="bold cyan")
    banner_text.append("║        欢迎使用交互式数据分析工作台！                         ║\n", style="bold green")
    banner_text.append("║        加载数据 → 清洗处理 → 探索分析 → 可视化               ║\n", style="bold yellow")
    banner_text.append("║        机器学习 → 生成报告                                   ║\n", style="bold yellow")
    banner_text.append("║                                                              ║\n", style="bold cyan")
    banner_text.append("╚══════════════════════════════════════════════════════════════╝\n", style="bold cyan")
    
    console.print(banner_text)
    
    features_text = """
[bold cyan]功能特性：[/bold cyan]
  • [green]数据加载[/green] - 支持 CSV 和 Excel 文件
  • [green]数据清洗[/green] - 缺失值处理、异常值检测、去重
  • [green]探索性分析[/green] - 描述性统计、相关性分析、分组统计
  • [green]可视化图表[/green] - 折线图、柱状图、饼图、热力图等
  • [green]机器学习[/green] - 分类、回归、聚类算法
  • [green]报告生成[/green] - 自动生成 Markdown 分析报告
    """
    console.print(Panel.fit(features_text, title="[bold]功能概览[/bold]", border_style="blue"))
    console.print()


def check_data_loaded() -> bool:
    """检查数据是否已加载"""
    if df is None or data_info is None:
        console.print("[bold red]⚠ 错误：请先加载数据！[/bold red]")
        return False
    return True


def load_data_menu():
    """数据加载菜单"""
    global df, data_info, current_file_path
    
    console.print(Panel.fit("[bold cyan]📁 数据加载[/bold cyan]", border_style="cyan"))
    
    while True:
        console.print("\n[bold yellow]请选择操作：[/bold yellow]")
        console.print("1. 加载数据文件")
        console.print("2. 预览数据")
        console.print("3. 查看数据信息")
        console.print("0. 返回主菜单")
        
        choice = Prompt.ask("[bold]请输入选项编号[/bold]", choices=["0", "1", "2", "3"], default="0")
        
        if choice == "0":
            return
        
        elif choice == "1":
            file_path = Prompt.ask("\n[bold]请输入数据文件路径[/bold]")
            
            if not os.path.exists(file_path):
                console.print(f"[bold red]✗ 文件不存在: {file_path}[/bold red]")
                continue
            
            try:
                with console.status("[bold yellow]正在加载数据...") as status:
                    df, data_info = loader.load_data(file_path)
                    current_file_path = file_path
                    
                console.print(f"[bold green]✓ 数据加载成功！[/bold green]")
                
            except Exception as e:
                console.print(f"[bold red]✗ 加载失败: {str(e)}[/bold red]")
        
        elif choice == "2":
            if not check_data_loaded():
                continue
            
            n = Prompt.ask("预览行数", default="5")
            try:
                n = int(n)
                loader.preview_data(df, n)
            except ValueError:
                console.print("[bold red]请输入有效的数字[/bold red]")
        
        elif choice == "3":
            if not check_data_loaded():
                continue
            
            show_data_info()


def show_data_info():
    """显示数据信息"""
    if not check_data_loaded():
        return
    
    console.print(Panel.fit("[bold cyan]数据概览[/bold cyan]", border_style="cyan"))
    
    table = Table(show_header=False, border_style="cyan")
    table.add_column("项目", style="cyan", width=20)
    table.add_column("值", style="yellow")
    
    table.add_row("文件路径", current_file_path or "N/A")
    table.add_row("数据形状", f"{data_info['shape'][0]} 行 × {data_info['shape'][1]} 列")
    table.add_row("内存占用", f"{data_info['memory_usage'] / 1024:.2f} KB")
    table.add_row("数值列", f"{len(data_info['numeric_columns'])} 个")
    table.add_row("分类列", f"{len(data_info['categorical_columns'])} 个")
    
    console.print(table)
    
    console.print("\n[bold cyan]列名列表：[/bold cyan]")
    for i, col in enumerate(data_info['columns'], 1):
        col_type = data_info['dtypes'].get(col, 'unknown')
        missing_count = data_info['missing'].get(col, 0)
        console.print(f"  {i}. [green]{col}[/green] ({col_type}) - 缺失: {missing_count}")


def cleaning_menu():
    """数据清洗菜单"""
    if not check_data_loaded():
        return
    
    while True:
        console.print(Panel.fit("[bold cyan]🧹 数据清洗[/bold cyan]", border_style="cyan"))
        
        console.print("\n[bold yellow]请选择操作：[/bold yellow]")
        console.print("1. 查看缺失值信息")
        console.print("2. 处理缺失值")
        console.print("3. 删除重复行")
        console.print("4. 检测异常值 (IQR)")
        console.print("5. 检测异常值 (Z-Score)")
        console.print("6. 转换数据类型")
        console.print("0. 返回主菜单")
        
        choice = Prompt.ask("[bold]请输入选项编号[/bold]", choices=["0", "1", "2", "3", "4", "5", "6"], default="0")
        
        if choice == "0":
            return
        
        elif choice == "1":
            cleaner.get_missing_info(df)
        
        elif choice == "2":
            handle_missing_values()
        
        elif choice == "3":
            handle_remove_duplicates()
        
        elif choice == "4":
            handle_outlier_detection("iqr")
        
        elif choice == "5":
            handle_outlier_detection("zscore")
        
        elif choice == "6":
            handle_dtype_conversion()


def handle_missing_values():
    """处理缺失值"""
    console.print("\n[bold]处理缺失值[/bold]")
    
    columns = data_info['columns']
    console.print("可用的列：")
    for i, col in enumerate(columns, 1):
        missing = data_info['missing'].get(col, 0)
        console.print(f"  {i}. {col} (缺失: {missing})")
    
    col_choice = Prompt.ask("\n选择要处理的列（逗号分隔，如 1,2,3）或输入 'all' 处理所有列")
    
    if col_choice.lower() == 'all':
        selected_cols = None
    else:
        try:
            indices = [int(x.strip()) for x in col_choice.split(',')]
            selected_cols = [columns[i-1] for i in indices if 1 <= i <= len(columns)]
        except ValueError:
            console.print("[bold red]输入格式错误[/bold red]")
            return
    
    console.print("\n[bold]选择填充策略：[/bold]")
    console.print("1. mean (均值填充) - 仅适用于数值列")
    console.print("2. median (中位数填充) - 仅适用于数值列")
    console.print("3. mode (众数填充)")
    console.print("4. forward_fill (前向填充)")
    console.print("5. backward_fill (后向填充)")
    console.print("6. drop (删除缺失行)")
    
    strategy_choice = Prompt.ask("选择策略", choices=["1", "2", "3", "4", "5", "6"])
    strategies = {
        "1": "mean",
        "2": "median", 
        "3": "mode",
        "4": "forward_fill",
        "5": "backward_fill",
        "6": "drop"
    }
    
    strategy = strategies[strategy_choice]
    
    try:
        df = cleaner.handle_missing_values(df, strategy=strategy, columns=selected_cols)
        
        data_info['missing'] = df.isnull().sum().to_dict()
        console.print("[bold green]✓ 缺失值处理完成！[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ 处理失败: {str(e)}[/bold red]")


def handle_remove_duplicates():
    """删除重复行"""
    global df, data_info
    
    original_shape = df.shape[0]
    df = cleaner.remove_duplicates(df)
    new_shape = df.shape[0]
    
    data_info['shape'] = df.shape
    
    if Confirm.ask("\n是否将清洗后的数据保存为新文件？"):
        default_name = current_file_path.replace('.csv', '_cleaned.csv') if current_file_path else 'cleaned_data.csv'
        save_path = Prompt.ask("输入保存路径", default=default_name)
        
        try:
            df.to_csv(save_path, index=False, encoding='utf-8')
            console.print(f"[bold green]✓ 数据已保存到: {save_path}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]✗ 保存失败: {str(e)}[/bold red]")


def handle_outlier_detection(method: str):
    """检测异常值"""
    global df, data_info
    
    numeric_cols = data_info['numeric_columns']
    
    if not numeric_cols:
        console.print("[bold yellow]没有数值列可供分析[/bold yellow]")
        return
    
    console.print("\n[bold]可用的数值列：[/bold]")
    for i, col in enumerate(numeric_cols, 1):
        console.print(f"  {i}. {col}")
    
    col_choice = Prompt.ask("\n选择要分析的列", choices=[str(i) for i in range(1, len(numeric_cols) + 1)])
    column = numeric_cols[int(col_choice) - 1]
    
    if method == "iqr":
        outliers_df, stats_df = cleaner.detect_outliers_iqr(df, column)
    else:
        threshold = Prompt.ask("Z-Score 阈值", default="3")
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = 3.0
        
        outliers_df, stats_df = cleaner.detect_outliers_zscore(df, column, threshold)
    
    if Confirm.ask("\n是否删除检测到的异常值？"):
        df = df.drop(outliers_df.index)
        data_info['shape'] = df.shape
        console.print(f"[bold green]✓ 已删除 {len(outliers_df)} 行异常值[/bold green]")


def handle_dtype_conversion():
    """转换数据类型"""
    global df, data_info
    
    columns = data_info['columns']
    
    console.print("\n[bold]可用的列：[/bold]")
    for i, col in enumerate(columns, 1):
        current_type = str(df[col].dtype)
        console.print(f"  {i}. {col} (当前类型: {current_type})")
    
    col_choice = Prompt.ask("\n选择要转换类型的列", 
                            choices=[str(i) for i in range(1, len(columns) + 1)])
    column = columns[int(col_choice) - 1]
    
    console.print("\n[bold]目标数据类型：[/bold]")
    console.print("1. int64 (整数)")
    console.print("2. float64 (浮点数)")
    console.print("3. str (字符串)")
    console.print("4. category (分类)")
    console.print("5. datetime (日期时间)")
    
    dtype_choice = Prompt.ask("选择目标类型", choices=["1", "2", "3", "4", "5"])
    dtype_map = {
        "1": "int64",
        "2": "float64",
        "3": "str",
        "4": "category",
        "5": "datetime"
    }
    
    target_dtype = dtype_map[dtype_choice]
    
    try:
        df = cleaner.convert_dtype(df, column, target_dtype)
        data_info['dtypes'][column] = str(df[column].dtype)
        console.print("[bold green]✓ 数据类型转换完成！[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ 转换失败: {str(e)}[/bold red]")


def eda_menu():
    """探索性数据分析菜单"""
    if not check_data_loaded():
        return
    
    global stats_results
    
    while True:
        console.print(Panel.fit("[bold cyan]📈 探索性数据分析 (EDA)[/bold cyan]", border_style="cyan"))
        
        console.print("\n[bold yellow]请选择操作：[/bold yellow]")
        console.print("1. 描述性统计")
        console.print("2. 相关性分析")
        console.print("3. 分组统计")
        console.print("4. 分布分析")
        console.print("0. 返回主菜单")
        
        choice = Prompt.ask("[bold]请输入选项编号[/bold]", choices=["0", "1", "2", "3", "4"], default="0")
        
        if choice == "0":
            return
        
        elif choice == "1":
            stats_df = eda.descriptive_stats(df)
            if not stats_df.empty:
                if stats_results is None:
                    stats_results = {}
                stats_results['describe'] = stats_df.to_dict()
                console.print("[bold green]✓ 描述性统计完成！[/bold green]")
        
        elif choice == "2":
            corr_matrix = eda.correlation_analysis(df)
            if not corr_matrix.empty:
                if stats_results is None:
                    stats_results = {}
                stats_results['correlation'] = corr_matrix.to_dict()
                console.print("[bold green]✓ 相关性分析完成！[/bold green]")
        
        elif choice == "3":
            handle_group_analysis()
        
        elif choice == "4":
            handle_distribution_analysis()


def handle_group_analysis():
    """分组统计"""
    columns = data_info['columns']
    
    console.print("\n[bold]选择分组列：[/bold]")
    for i, col in enumerate(columns, 1):
        console.print(f"  {i}. {col}")
    
    group_choice = Prompt.ask("输入分组列编号", 
                              choices=[str(i) for i in range(1, len(columns) + 1)])
    group_col = columns[int(group_choice) - 1]
    
    console.print("\n[bold]选择聚合列（必须是数值列）：[/bold]")
    numeric_cols = data_info['numeric_columns']
    for i, col in enumerate(numeric_cols, 1):
        console.print(f"  {i}. {col}")
    
    if not numeric_cols:
        console.print("[bold yellow]没有数值列可供聚合[/bold yellow]")
        return
    
    agg_choice = Prompt.ask("输入聚合列编号", 
                          choices=[str(i) for i in range(1, len(numeric_cols) + 1)])
    agg_col = numeric_cols[int(agg_choice) - 1]
    
    console.print("\n[bold]选择聚合函数：[/bold]")
    console.print("1. mean (均值)")
    console.print("2. sum (求和)")
    console.print("3. count (计数)")
    console.print("4. min (最小值)")
    console.print("5. max (最大值)")
    console.print("6. std (标准差)")
    console.print("7. median (中位数)")
    
    func_choice = Prompt.ask("选择聚合函数", 
                            choices=["1", "2", "3", "4", "5", "6", "7"])
    func_map = {
        "1": "mean", "2": "sum", "3": "count",
        "4": "min", "5": "max", "6": "std", "7": "median"
    }
    agg_func = func_map[func_choice]
    
    result = eda.group_analysis(df, group_col, agg_col, agg_func)
    
    if not result.empty and Confirm.ask("\n是否保存分组结果？"):
        default_name = f"group_{group_col}_{agg_col}.csv"
        save_path = Prompt.ask("输入保存路径", default=default_name)
        try:
            result.to_csv(save_path, index=False)
            console.print(f"[bold green]✓ 结果已保存到: {save_path}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]✗ 保存失败: {str(e)}[/bold red]")


def handle_distribution_analysis():
    """分布分析"""
    numeric_cols = data_info['numeric_columns']
    
    if not numeric_cols:
        console.print("[bold yellow]没有数值列可供分析[/bold yellow]")
        return
    
    console.print("\n[bold]可用的数值列：[/bold]")
    for i, col in enumerate(numeric_cols, 1):
        console.print(f"  {i}. {col}")
    
    col_choice = Prompt.ask("选择要分析的列", 
                            choices=[str(i) for i in range(1, len(numeric_cols) + 1)])
    column = numeric_cols[int(col_choice) - 1]
    
    dist_info = eda.distribution_analysis(df, column)
    
    if dist_info and Confirm.ask("\n是否将此分布信息保存？"):
        if stats_results is None:
            stats_results = {}
        stats_results['distribution'] = stats_results.get('distribution', {})
        stats_results['distribution'][column] = dist_info
        console.print("[bold green]✓ 分布信息已保存！[/bold green]")


def visualization_menu():
    """可视化菜单"""
    if not check_data_loaded():
        return
    
    global charts_generated
    
    chart_types = {
        "1": ("折线图", visualizer.plot_line, ["x", "y"]),
        "2": ("柱状图", visualizer.plot_bar, ["x", "y"]),
        "3": ("饼图", visualizer.plot_pie, ["column"]),
        "4": ("热力图", visualizer.plot_heatmap, []),
        "5": ("箱线图", visualizer.plot_box, ["column"]),
        "6": ("散点图", visualizer.plot_scatter, ["x", "y"]),
        "7": ("直方图", visualizer.plot_histogram, ["column"]),
    }
    
    while True:
        console.print(Panel.fit("[bold cyan]📊 可视化图表[/bold cyan]", border_style="cyan"))
        
        console.print("\n[bold yellow]支持的图表类型：[/bold yellow]")
        for key, (name, _, _) in chart_types.items():
            console.print(f"  {key}. {name}")
        console.print("  0. 返回主菜单")
        
        choice = Prompt.ask("\n[bold]请选择图表类型[/bold]", 
                           choices=["0", "1", "2", "3", "4", "5", "6", "7"])
        
        if choice == "0":
            return
        
        chart_name, plot_func, required_cols = chart_types[choice]
        handle_chart_generation(chart_name, plot_func, required_cols)


def handle_chart_generation(chart_name: str, plot_func, required_cols):
    """处理图表生成"""
    global charts_generated
    
    console.print(f"\n[bold]生成 {chart_name}[/bold]")
    
    kwargs = {}
    
    if "x" in required_cols:
        columns = data_info['columns']
        console.print("\n[bold]选择 X 轴列：[/bold]")
        for i, col in enumerate(columns, 1):
            console.print(f"  {i}. {col}")
        x_choice = Prompt.ask("输入列编号", choices=[str(i) for i in range(1, len(columns) + 1)])
        kwargs['x'] = columns[int(x_choice) - 1]
    
    if "y" in required_cols:
        numeric_cols = data_info['numeric_columns']
        console.print("\n[bold]选择 Y 轴列（数值列）：[/bold]")
        for i, col in enumerate(numeric_cols, 1):
            console.print(f"  {i}. {col}")
        y_choice = Prompt.ask("输入列编号", choices=[str(i) for i in range(1, len(numeric_cols) + 1)])
        kwargs['y'] = numeric_cols[int(y_choice) - 1]
    
    if "column" in required_cols:
        columns = data_info['columns']
        console.print("\n[bold]选择列：[/bold]")
        for i, col in enumerate(columns, 1):
            console.print(f"  {i}. {col}")
        col_choice = Prompt.ask("输入列编号", choices=[str(i) for i in range(1, len(columns) + 1)])
        kwargs['column'] = columns[int(col_choice) - 1]
    
    title = Prompt.ask("输入图表标题", default=f"{chart_name} - 数据可视化")
    filename = Prompt.ask("输入保存文件名（留空自动生成）", default="")
    
    if filename.strip() == "":
        filename = None
    else:
        if not filename.endswith('.png'):
            filename += '.png'
    
    kwargs['title'] = title
    
    if filename:
        kwargs['filename'] = filename
    
    try:
        if chart_name == "热力图":
            filepath = plot_func(df, **kwargs)
        else:
            filepath = plot_func(df, **kwargs)
        
        charts_generated.append(filepath)
        console.print(f"[bold green]✓ 图表已保存到: {filepath}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ 生成失败: {str(e)}[/bold red]")
    
    if Confirm.ask("\n继续生成其他图表？"):
        visualization_menu()


def machine_learning_menu():
    """机器学习菜单"""
    if not check_data_loaded():
        return
    
    algorithms = {
        "1": ("逻辑回归分类", modeler.train_logistic_regression, True),
        "2": ("随机森林分类", modeler.train_random_forest, True),
        "3": ("线性回归", modeler.train_linear_regression, False),
        "4": ("K-Means 聚类", modeler.train_kmeans, False),
    }
    
    while True:
        console.print(Panel.fit("[bold cyan]🤖 机器学习[/bold cyan]", border_style="cyan"))
        
        console.print("\n[bold yellow]支持的算法：[/bold yellow]")
        for key, (name, _, needs_y) in algorithms.items():
            algo_type = "(分类/回归)" if needs_y else "(聚类)"
            console.print(f"  {key}. {name} {algo_type}")
        console.print("  0. 返回主菜单")
        
        choice = Prompt.ask("\n[bold]请选择算法[/bold]", 
                           choices=["0", "1", "2", "3", "4"])
        
        if choice == "0":
            return
        
        algo_name, train_func, needs_target = algorithms[choice]
        handle_ml_training(algo_name, train_func, needs_target)


def handle_ml_training(algo_name: str, train_func, needs_target: bool):
    """处理机器学习训练"""
    global stats_results
    
    console.print(f"\n[bold]训练 {algo_name} 模型[/bold]")
    
    numeric_cols = data_info['numeric_columns']
    
    if not numeric_cols:
        console.print("[bold red]没有数值列可用于建模[/bold red]")
        return
    
    console.print("\n[bold]可用的数值特征列：[/bold]")
    for i, col in enumerate(numeric_cols, 1):
        console.print(f"  {i}. {col}")
    
    feature_indices = Prompt.ask("\n选择特征列（逗号分隔，如 1,2,3）").split(',')
    try:
        feature_cols = [numeric_cols[int(x.strip()) - 1] for x in feature_indices]
    except (ValueError, IndexError):
        console.print("[bold red]输入格式错误[/bold red]")
        return
    
    X = df[feature_cols].values
    
    y = None
    if needs_target:
        console.print("\n[bold]选择目标列：[/bold]")
        target_cols = [col for col in numeric_cols if col not in feature_cols]
        
        if not target_cols:
            console.print("[bold yellow]没有可用的目标列[/bold yellow]")
            return
        
        for i, col in enumerate(target_cols, 1):
            console.print(f"  {i}. {col}")
        
        target_choice = Prompt.ask("输入目标列编号", 
                                  choices=[str(i) for i in range(1, len(target_cols) + 1)])
        target_col = target_cols[int(target_choice) - 1]
        y = df[target_col].values
        
        test_size = Prompt.ask("测试集比例", default="0.2")
        try:
            test_size = float(test_size)
        except ValueError:
            test_size = 0.2
    
    try:
        with console.status(f"[bold yellow]正在训练 {algo_name} 模型...") as status:
            if needs_target:
                if "随机森林" in algo_name:
                    result = train_func(X, y, n_estimators=100, test_size=test_size)
                else:
                    result = train_func(X, y, test_size=test_size)
            else:
                n_clusters = Prompt.ask("聚类数量", default="4")
                try:
                    n_clusters = int(n_clusters)
                except ValueError:
                    n_clusters = 4
                result = train_func(X, n_clusters=n_clusters)
        
        display_ml_results(algo_name, result)
        
        if Confirm.ask("\n是否保存模型训练结果？"):
            if stats_results is None:
                stats_results = {}
            stats_results['ml_models'] = stats_results.get('ml_models', {})
            stats_results['ml_models'][algo_name] = {k: v for k, v in result.items() if k != 'model'}
            console.print("[bold green]✓ 训练结果已保存！[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ 训练失败: {str(e)}[/bold red]")


def display_ml_results(algo_name: str, result: dict):
    """显示机器学习结果"""
    console.print(f"\n[bold green]✓ {algo_name} 训练完成！[/bold green]")
    
    table = Table(title=f"{algo_name} 训练结果", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan")
    table.add_column("值", justify="right", style="yellow")
    
    if 'accuracy' in result:
        table.add_row("准确率 (Accuracy)", f"{result['accuracy']:.4f}")
    
    if 'r2_score' in result:
        table.add_row("R² 分数", f"{result['r2_score']:.4f}")
    
    if 'silhouette_score' in result:
        table.add_row("轮廓系数", f"{result['silhouette_score']:.4f}")
    
    if 'report' in result:
        report = result['report']
        if isinstance(report, dict):
            if 'macro avg' in report:
                table.add_row("宏平均精确率", f"{report['macro avg']['precision']:.4f}")
                table.add_row("宏平均召回率", f"{report['macro avg']['recall']:.4f}")
                table.add_row("宏平均 F1", f"{report['macro avg']['f1-score']:.4f}")
    
    console.print(table)


def report_menu():
    """报告生成菜单"""
    if not check_data_loaded():
        return
    
    while True:
        console.print(Panel.fit("[bold cyan]📝 报告生成[/bold cyan]", border_style="cyan"))
        
        console.print("\n[bold yellow]请选择操作：[/bold yellow]")
        console.print("1. 生成完整报告")
        console.print("2. 自定义报告内容")
        console.print("3. 查看已保存的图表")
        console.print("0. 返回主菜单")
        
        choice = Prompt.ask("[bold]请输入选项编号[/bold]", choices=["0", "1", "2", "3"], default="0")
        
        if choice == "0":
            return
        
        elif choice == "1":
            generate_full_report()
        
        elif choice == "2":
            generate_custom_report()
        
        elif choice == "3":
            view_saved_charts()


def generate_full_report():
    """生成完整报告"""
    console.print("\n[bold]生成完整数据分析报告[/bold]")
    
    title = Prompt.ask("报告标题", default="数据分析报告")
    
    info_for_report = {
        'file_path': current_file_path,
        'list_shape': list(data_info['shape']),
        'columns': data_info['columns'],
        'dtypes': data_info['dtypes'],
    }
    
    conclusions = Prompt.ask("分析结论（可选，直接回车跳过）", default="")
    
    console.print("\n[bold yellow]正在生成报告...[/bold yellow]")
    
    try:
        report_path = generator.generate_report(
            title=title,
            data_info=info_for_report,
            stats=stats_results,
            charts=charts_generated,
            conclusions=conclusions if conclusions else ""
        )
        
        console.print(f"[bold green]✓ 报告已生成: {report_path}[/bold green]")
        
        if Confirm.ask("是否打开报告文件？"):
            import webbrowser
            webbrowser.open(f'file://{report_path}')
            
    except Exception as e:
        console.print(f"[bold red]✗ 报告生成失败: {str(e)}[/bold red]")


def generate_custom_report():
    """生成自定义报告"""
    console.print("\n[bold]自定义报告内容[/bold]")
    
    title = Prompt.ask("报告标题", default="自定义分析报告")
    
    console.print("\n[bold]选择要包含的内容：[/bold yellow]")
    include_stats = Confirm.ask("包含统计摘要？", default=True)
    include_charts = Confirm.ask("包含图表引用？", default=True)
    conclusions = Prompt.ask("分析结论（可选）", default="")
    
    info_for_report = {
        'file_path': current_file_path,
        'list_shape': list(data_info['shape']),
        'columns': data_info['columns'],
        'dtypes': data_info['dtypes'],
    }
    
    stats = stats_results if include_stats else None
    charts = charts_generated if include_charts else []
    
    try:
        report_path = generator.generate_report(
            title=title,
            data_info=info_for_report,
            stats=stats,
            charts=charts,
            conclusions=conclusions if conclusions else ""
        )
        
        console.print(f"[bold green]✓ 报告已生成: {report_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ 报告生成失败: {str(e)}[/bold red]")


def view_saved_charts():
    """查看已保存的图表"""
    if not charts_generated:
        console.print("[bold yellow]还没有生成任何图表[/bold yellow]")
        return
    
    console.print(Panel.fit("[bold cyan]已保存的图表列表[/bold cyan]", border_style="cyan"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("序号", justify="right", style="cyan")
    table.add_column("文件路径", style="green")
    
    for i, chart_path in enumerate(charts_generated, 1):
        table.add_row(str(i), chart_path)
    
    console.print(table)


def show_status():
    """显示当前状态"""
    console.print(Panel.fit("[bold cyan]📋 当前状态[/bold cyan]", border_style="cyan"))
    
    table = Table(show_header=False, border_style="cyan")
    table.add_column("项目", style="cyan", width=25)
    table.add_column("状态", style="yellow")
    
    table.add_row("数据加载", "✓ 已加载" if df is not None else "✗ 未加载")
    
    if df is not None:
        table.add_row("数据形状", f"{df.shape[0]} 行 × {df.shape[1]} 列")
        table.add_row("文件路径", current_file_path or "N/A")
    
    if stats_results:
        table.add_row("统计结果", f"✓ 已生成 {len(stats_results)} 项")
    else:
        table.add_row("统计结果", "✗ 未生成")
    
    table.add_row("已生成图表", f"{len(charts_generated)} 个")
    
    console.print(table)


def main_menu():
    """主菜单"""
    while True:
        console.print("\n")
        show_status()
        
        console.print(Panel.fit("[bold cyan]🖥️ 主菜单[/bold cyan]", border_style="cyan"))
        
        console.print("\n[bold yellow]请选择操作：[/bold yellow]")
        console.print("1. 📁 数据加载")
        console.print("2. 🧹 数据清洗")
        console.print("3. 📈 探索性分析")
        console.print("4. 📊 可视化图表")
        console.print("5. 🤖 机器学习")
        console.print("6. 📝 生成报告")
        console.print("0. 🚪 退出程序")
        
        choice = Prompt.ask("\n[bold]请输入选项编号[/bold]", 
                          choices=["0", "1", "2", "3", "4", "5", "6"])
        
        if choice == "0":
            console.print("\n[bold cyan]感谢使用数据分析工具，再见！[/bold cyan]")
            break
        
        elif choice == "1":
            load_data_menu()
        
        elif choice == "2":
            cleaning_menu()
        
        elif choice == "3":
            eda_menu()
        
        elif choice == "4":
            visualization_menu()
        
        elif choice == "5":
            machine_learning_menu()
        
        elif choice == "6":
            report_menu()


def main():
    """主函数"""
    show_welcome()
    
    if not Confirm.ask("\n[bold]是否开始使用数据分析工具？[/bold]", default=True):
        console.print("[bold cyan]再见！[/bold cyan]")
        return
    
    main_menu()


if __name__ == "__main__":
    main()
