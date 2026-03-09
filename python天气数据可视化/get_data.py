#导入库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from crawl4ai import AsyncWebCrawler, BrowserConfig
from bs4 import BeautifulSoup
import json
import csv
import asyncio
import re
import os

#构建函数获取数据
async def get_data_and_wash():
    #设置浏览器配置
    browser_config = BrowserConfig(
    headless=True,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    # 从URL获取数据（7天预报页面）
    url = 'https://www.weather.com.cn/weather/101070202.shtml'
    #获取数据
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            print('请求成功')
            html = result.html
    except Exception as e:
        print(f"请求失败: {e}")
        return None
    #解析HTML
    soup = BeautifulSoup(html, 'html.parser')
    # 提取24小时详细天气数据
    print("\n=== 提取24小时详细天气数据 ===")
    # 在HTML中查找observe24h_data
    pattern = r'var observe24h_data = (\{.*?\});'
    match = re.search(pattern, html, re.DOTALL)
    # 初始化24小时数据列表
    hourly_data_list = []
    #检测数据，json转换
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            # 提取od2数组中的数据
            if 'od' in data and 'od2' in data['od']:
                for item in data['od']['od2']:
                    hourly_data = {
                        'time': item.get('od21', ''),
                        'temperature': item.get('od22', ''),
                        'wind_direction': item.get('od24', ''),
                        'wind_level': item.get('od25', ''),
                        'humidity': item.get('od27', '')
                    }
                    hourly_data_list.append(hourly_data)
                    print(f"提取时间: {hourly_data['time']}时, 温度: {hourly_data['temperature']}摄氏度, 风向: {hourly_data['wind_direction']}, 风力: {hourly_data['wind_level']}级, 湿度: {hourly_data['humidity']}%")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
    else:
        print("未找到observe24h_data数据")
    if hourly_data_list:
        print(f"成功提取 {len(hourly_data_list)} 条24小时数据")
    # 提取7天天气数据
    print("\n=== 提取7天天气数据 ===")
    weekly_data = soup.find_all('ul', class_='t clearfix')
    #初始化7天天气数据列表
    weekly_weather_list = []
    # 遍历每个ul元素
    for ul in weekly_data:
        li_elements = ul.find_all('li')
        for li in li_elements:
            date = li.find('h1').get_text().strip()
            weather = li.find('p', class_='wea').get_text().strip()
            
            # 提取温度
            temp_p = li.find('p', class_='tem')
            max_temp = temp_p.find('span').get_text().strip() if temp_p.find('span') else ''
            min_temp = temp_p.find('i').get_text().strip() if temp_p.find('i') else ''
            
            # 提取风向 - 从span标签的title属性中获取
            wind_spans = li.find('p', class_='win').find_all('span')
            wind_directions = [span.get('title', '') for span in wind_spans]
            wind_direction = '转'.join(wind_directions) if wind_directions else ''
            
            # 提取风力等级
            wind_level = li.find('p', class_='win').find('i').get_text().strip() if li.find('p', class_='win').find('i') else ''
            
            print(f"日期: {date}")
            print(f"天气: {weather}")
            print(f"最高温: {max_temp}℃")
            print(f"最低温: {min_temp}")
            print(f"风向: {wind_direction}")
            print(f"风力: {wind_level}")
            print("-" * 50)
            
            weekly_weather_list.append({
                'date': date,
                'weather': weather,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'wind_direction': wind_direction,
                'wind_level': wind_level
            })
    # 构建绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用绝对路径读取CSV文件
    csv_path = os.path.join(script_dir, 'hourly_weather_data.csv')
    #将提取到的天气数据保存到CSV文件
    if weekly_weather_list:
        csv_file = os.path.join(script_dir, 'weekly_weather_data.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['date', 'weather', 'max_temp', 'min_temp', 'wind_direction', 'wind_level']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(weekly_weather_list)
    if hourly_data_list:
        csv_file = os.path.join(script_dir, 'hourly_weather_data.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'temperature', 'wind_direction', 'wind_level', 'humidity']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hourly_data_list)
        print(f"\n数据已保存到 {csv_file}")
        print(f"共提取 {len(hourly_data_list)} 条24小时数据")
    if weekly_weather_list:
        csv_file = os.path.join(script_dir, 'weekly_weather_data.csv') 
        print(f"\n数据已保存到 {csv_file}")
        print(f"共提取 {len(weekly_weather_list)} 天的数据")
asyncio.run(get_data_and_wash())