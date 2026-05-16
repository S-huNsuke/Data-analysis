# 天气数据可视化

## 项目简介
基于Python的天气数据可视化工具，从中国天气网获取天气数据并进行可视化展示。

## 功能特性
- **数据获取**：自动从中国天气网获取24小时详细天气数据和7天天气预报
- **数据处理**：对获取的数据进行清洗和整理
- **24小时可视化**：
  - 24小时温度趋势折线图
  - 24小时湿度趋势折线图
  - 24小时平均风力风向雷达图
  - 24小时温湿度相关性散点图
- **7天可视化**：
  - 7天温度趋势折线图
  - 7天湿度趋势折线图
  - 7天平均风力风向雷达图

## 技术栈
- requests, crawl4ai
- pandas, numpy
- matplotlib, seaborn
- BeautifulSoup4

## 项目结构
```
weather-visualization/
├── get_data.py                # 天气数据获取脚本
├── hourly_plt.py              # 24小时数据可视化脚本
├── week_plt.py                # 7天数据可视化脚本
├── hourly_weather_data.csv    # 24小时天气数据（自动生成）
├── weekly_weather_data.csv    # 7天天气数据（自动生成）
└── output/                    # 输出结果
```

## 使用方法

```bash
# 安装依赖
uv sync

# 获取天气数据
uv run python weather-visualization/get_data.py

# 生成24小时可视化图表
uv run python weather-visualization/hourly_plt.py

# 生成7天可视化图表
uv run python weather-visualization/week_plt.py
```

输出结果保存在 `output/` 目录下。
