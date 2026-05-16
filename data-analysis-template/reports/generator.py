"""报告生成模块 - 生成 Markdown 格式的数据分析报告"""

import os
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
    """
    生成 Markdown 格式的数据分析报告
    
    Args:
        title: 报告标题
        data_info: 数据概览信息字典，包含 keys/list_shape/columns 等
        stats: 统计摘要字典，包含基本统计量等
        charts: 可视化图表路径列表
        conclusions: 分析结论文本
        output_path: 自定义输出路径，默认保存到 output/reports/
    
    Returns:
        str: 报告文件的绝对路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    default_filename = f"{safe_title}_{timestamp}.md"
    
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "output" / "reports"
    else:
        output_dir = Path(output_path).parent
        default_filename = Path(output_path).name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / default_filename
    
    md_content = _build_markdown(title, data_info, stats, charts, conclusions)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return str(report_path.resolve())


def _build_markdown(
    title: str,
    data_info: dict,
    stats: dict = None,
    charts: list = None,
    conclusions: str = ""
) -> str:
    """构建 Markdown 报告内容"""
    lines = []
    
    lines.append(f"# {title}\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    
    lines.extend(_build_data_overview(data_info))
    
    if stats:
        lines.extend(_build_statistics(stats))
    
    if charts:
        lines.extend(_build_visualizations(charts))
    
    if conclusions:
        lines.extend(_build_conclusions(conclusions))
    
    return "\n".join(lines)


def _build_data_overview(data_info: dict) -> list:
    """构建数据概览章节"""
    lines = ["## 数据概览\n"]
    
    if 'file_path' in data_info:
        lines.append(f"- **文件路径**: `{data_info['file_path']}`\n")
    
    if 'list_shape' in data_info:
        shape = data_info['list_shape']
        lines.append(f"- **数据形状**: {shape[0]} 行 × {shape[1]} 列\n")
    
    if 'row_count' in data_info:
        lines.append(f"- **行数**: {data_info['row_count']}\n")
    
    if 'column_count' in data_info:
        lines.append(f"- **列数**: {data_info['column_count']}\n")
    
    if 'columns' in data_info:
        lines.append("\n**列名列表**:\n")
        for col in data_info['columns']:
            col_type = data_info.get('dtypes', {}).get(col, 'unknown')
            lines.append(f"- `{col}` ({col_type})\n")
    
    if 'keys' in data_info:
        lines.append(f"\n**数据键/索引**: {', '.join(data_info['keys'])}\n")
    
    lines.append("\n---\n")
    return lines


def _build_statistics(stats: dict) -> list:
    """构建统计摘要章节"""
    lines = ["## 统计摘要\n"]
    
    if 'describe' in stats:
        lines.append("\n### 数值列统计\n")
        lines.append("| 列名 | 计数 | 均值 | 标准差 | 最小值 | 25% | 50% | 75% | 最大值 |\n")
        lines.append("|------|------|------|--------|--------|-----|-----|-----|--------|\n")
        
        for col, desc in stats['describe'].items():
            row = [
                col,
                f"{desc.get('count', 0):.0f}",
                f"{desc.get('mean', 0):.2f}",
                f"{desc.get('std', 0):.2f}",
                f"{desc.get('min', 0):.2f}",
                f"{desc.get('25%', 0):.2f}",
                f"{desc.get('50%', 0):.2f}",
                f"{desc.get('75%', 0):.2f}",
                f"{desc.get('max', 0):.2f}"
            ]
            lines.append("| " + " | ".join(row) + " |\n")
    
    if 'missing' in stats:
        lines.append("\n### 缺失值统计\n")
        for col, count in stats['missing'].items():
            pct = (count / stats.get('total_rows', 1)) * 100
            lines.append(f"- {col}: {count} ({pct:.1f}%)\n")
    
    if 'categorical' in stats:
        lines.append("\n### 分类列统计\n")
        for col, value_counts in stats['categorical'].items():
            lines.append(f"\n**{col}**:\n")
            for val, count in list(value_counts.items())[:10]:
                lines.append(f"- {val}: {count}\n")
            if len(value_counts) > 10:
                lines.append(f"- ... (共 {len(value_counts)} 个类别)\n")
    
    lines.append("\n---\n")
    return lines


def _build_visualizations(charts: list) -> list:
    """构建可视化图表章节"""
    lines = ["## 可视化图表\n"]
    
    for i, chart_path in enumerate(charts, 1):
        chart_name = Path(chart_path).stem.replace('_', ' ').title()
        lines.append(f"### 图 {i}: {chart_name}\n")
        lines.append(f"![{chart_name}]({chart_path})\n")
        lines.append(f"*图表路径: `{chart_path}`*\n")
    
    lines.append("\n---\n")
    return lines


def _build_conclusions(conclusions: str) -> list:
    """构建分析结论章节"""
    lines = ["## 分析结论\n"]
    lines.append(f"{conclusions}\n")
    return lines
