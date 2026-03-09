# 天气数据可视化项目

## 项目简介
本项目是一个基于Python的天气数据可视化工具，用于从中国天气网获取天气数据并进行可视化展示。

## 功能特性
1. **数据获取**：自动从中国天气网获取24小时详细天气数据和7天天气预报
2. **数据处理**：对获取的数据进行清洗和整理
3. **可视化展示**：
   - 24小时温度趋势折线图
   - 24小时湿度趋势折线图
   - 24小时平均风力风向雷达图
   - 24小时温湿度相关性散点图
   - 7天温度趋势折线图
   - 7天湿度趋势折线图
   - 7天平均风力风向雷达图

## 技术栈
- **数据获取**：requests, crawl4ai
- **数据处理**：pandas, numpy
- **数据可视化**：matplotlib, seaborn
- **HTML解析**：BeautifulSoup4
- **其他**：re, json, csv, asyncio, os

## 项目结构
```
python天气数据可视化/
├── get_data.py           # 天气数据获取脚本
├── hourly_plt.py         # 24小时数据可视化脚本
├── weekly_plt.py         # 7天数据可视化脚本
├── hourly_weather_data.csv  # 24小时天气数据（自动生成）
├── weekly_weather_data.csv  # 7天天气数据（自动生成）
└── README.md             # 项目说明文档
```

## 使用方法

### 1. 安装依赖
```bash
pip install pandas matplotlib seaborn requests crawl4ai beautifulsoup4 numpy
```

### 2. 获取天气数据
运行数据获取脚本：
```bash
python get_data.py
```
该脚本会从中国天气网获取当地的天气数据，并保存为CSV文件。

### 3. 生成可视化图表
运行24小时数据可视化脚本：
```bash
python hourly_plt.py
```
该脚本会读取CSV文件中的数据，生成并保存以下图表：
- 24小时温度趋势图 (24weather_trend.png/jpg)
- 24小时湿度趋势图 (24hourly_hum_trend.png/jpg)
- 24小时平均风力风向雷达图 (wind_radar.png/jpg)
- 24小时温湿度相关性图 (24hourly_temp_hum.png/jpg)

运行7天数据可视化脚本：
```bash
python weekly_plt.py
```
该脚本会读取CSV文件中的数据，生成并保存以下图表：
- 7天温度趋势图 (7weather_trend.png/jpg)
- 7天湿度趋势图 (7weekly_hum_trend.png/jpg)
- 7天平均风力风向雷达图 (7wind_radar.png/jpg)

## 文件说明

### get_data.py
- 使用异步网页爬虫获取天气数据
- 解析HTML提取24小时详细天气数据和7天天气预报
- 将数据保存为CSV文件

### hourly_plt.py
- 读取CSV文件中的24小时天气数据
- 绘制各种可视化图表
- 保存图表为PNG和JPG格式

### weekly_plt.py
- 读取CSV文件中的7天天气数据
- 绘制各种可视化图表
- 保存图表为PNG和JPG格式

## 注意事项
1. 确保网络连接正常，以便能够获取天气数据
2. 项目默认获取大连（101070202）的天气数据，如需获取其他城市数据，请修改get_data.py中的URL
3. 图表会保存在项目目录下

## 扩展建议
1. 添加更多城市的天气数据获取功能
2. 实现定时自动获取和更新数据
3. 添加更多可视化图表类型
4. 开发Web界面展示数据和图表

## 许可证
本项目采用MIT许可证。