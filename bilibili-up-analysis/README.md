# B站UP主数据可视化分析

## 项目简介
基于Python的B站UP主数据可视化分析项目，通过对B站新榜数据的分析，生成多种可视化图表，帮助用户直观了解B站UP主的各项数据指标和分布情况。

## 功能特性
- **数据处理**：读取B站新榜Excel数据，自动清理和转换数据格式（如处理"w"、"亿"等单位）
- **性别分布分析**：统计不同性别UP主的分布情况
- **获赞/粉丝/投币数分析**：UP主获赞数、粉丝数、投币数的整体情况和分布特征
- **分类树状图**：以树状图展示各类型UP主的获赞数、投币数、视频数量分布
- **词云分析**：生成UP主标签的词云图，展示热门标签

## 技术栈
- pandas, numpy
- matplotlib, seaborn
- pyecharts
- wordcloud, jieba
- PIL (Pillow)

## 项目结构
```
bilibili-up-analysis/
├── main.py                        # 主程序入口
├── B站新榜_综合指数榜单.xlsx       # B站UP主数据文件
├── cn_stopwords.txt               # 中文停用词表
├── bilibili.jpg                   # 词云背景图片
└── output/                        # 输出结果
```

## 使用方法

```bash
# 安装依赖
uv sync

# 运行分析程序
uv run python bilibili-up-analysis/main.py
```

输出结果保存在 `output/` 目录下。
