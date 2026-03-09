# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# 中文字符设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据并进行数据处理
def data_wash():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用绝对路径读取CSV文件
    csv_path = os.path.join(script_dir, 'weekly_weather_data.csv')
    df_week = pd.read_csv(csv_path)
    df_week['min_temp'] = df_week['min_temp'].astype(str).str.split('℃').str[0].astype(float)
    df_week['max_temp'] = df_week['max_temp'].astype(str).str.split('℃').str[0].astype(float)
    df_week['wind_level'] = df_week['wind_level'].str.extract(r'(\d+)').astype(int)
    df_week[['wind1', 'wind2']] = df_week['wind_direction'].str.split('转', expand=True)
    df_week['date'] = df_week['date'].str.extract(r'(\d+)').astype(int)
    df_week['date'] = pd.to_datetime('2025-01-' + df_week['date'].astype(str).str.zfill(2), format='%Y-%m-%d')
    print(df_week.head(7))
    print('数据处理完成')
    return df_week, script_dir
df_week, script_dir = data_wash()

#绘制天气变化趋势图
def plt_week(df, dir_path):
    # 计算统计数据
    average_max_temp = df['max_temp'].mean()
    average_min_temp = df['min_temp'].mean()
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='date', y='min_temp', data=df, label='最低气温', color='#1f77b4', linewidth=2.5, marker='o', markersize=8)
    sns.lineplot(x='date', y='max_temp', data=df, label='最高气温', color='#ff7f0e', linewidth=2.5, marker='o', markersize=8)
    
    plt.axhline(average_min_temp, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axhline(average_max_temp, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.title('7日温度趋势图', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('日期', fontsize=12, fontweight='bold')
    plt.ylabel('温度 (℃)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
    plt.gca().set_facecolor('#f8f9fa')
    
    for i, row in df.iterrows():
        plt.text(row['date'], row['min_temp'] - 1.2, f"{row['min_temp']:.0f}", 
                ha='center', va='top', fontsize=10, color='#1f77b4', zorder=10,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#1f77b4', alpha=0.9, linewidth=1.2))
        plt.text(row['date'], row['max_temp'] + 1.2, f"{row['max_temp']:.0f}", 
                ha='center', va='bottom', fontsize=10, color='#ff7f0e', zorder=10,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#ff7f0e', alpha=0.9, linewidth=1.2))
    
    plt.text(df['date'].iloc[0], average_min_temp, f'平均最低温: {average_min_temp:.1f}℃', 
            ha='left', va='bottom', fontsize=10, color='#1f77b4', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1f77b4', alpha=0.9, linewidth=1.2))
    plt.text(df['date'].iloc[0], average_max_temp, f'平均最高温: {average_max_temp:.1f}℃', 
            ha='left', va='top', fontsize=10, color='#ff7f0e', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ff7f0e', alpha=0.9, linewidth=1.2))
    
    ax = plt.gca()
    y_min = df['min_temp'].min() - 3
    y_max = df['max_temp'].max() + 10
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    # 使用绝对路径保存图片
    plt.savefig(os.path.join(dir_path, 'weather_trend.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_path, 'weather_trend.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    print('图片已保存为 weather_trend.png 和 weather_trend.jpg')
    plt.show()
plt_week(df_week, script_dir)

#绘制风力风向雷达图
def wind_switch(wind):
    if pd.isna(wind) or wind in ['无持续风向', 'None', '']:
        return None  # 忽略无持续风向的数据
    elif wind == '北风':
        return 0
    elif wind == '东北风':
        return 45
    elif wind == '东风':
        return 90
    elif wind == '东南风':
        return 135
    elif wind == '南风':
        return 180
    elif wind == '西南风':
        return 225
    elif wind == '西风':
        return 270
    elif wind == '西北风':
        return 315
    else:
        return None  # 忽略未知风向
def wind_radar(df, dir_path):
    # 确保使用所有天的数据（根据实际数据行数）
    days = len(df)
    print(f"处理 {days} 天的天气数据")
    
    wind1 = list(df['wind1'])
    wind2 = list(df['wind2'])
    wind_speed = list(df['wind_level'])
    
    # 将风向转换为角度，忽略无效值
    wind1_deg = []
    for w in wind1:
        deg = wind_switch(w)
        if deg is not None:
            wind1_deg.append(deg)
        else:
            wind1_deg.append(-1)  # 使用-1标记无效风向
    
    wind2_deg = []
    for w in wind2:
        deg = wind_switch(w)
        if deg is not None:
            wind2_deg.append(deg)
        else:
            wind2_deg.append(-1)  # 使用-1标记无效风向
    
    # 定义8个风向及其正确角度 (0° = 北风)
    directions = ['北风', '东北风', '东风', '东南风', '南风', '西南风', '西风', '西北风']
    degs = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # 每个方向的正确角度
    temp = []
    
    for deg in degs:
        speed = []
        for i in range(days):  # 检查所有天的数据
            if wind1_deg[i] == deg:
                speed.append(wind_speed[i])
            if wind2_deg[i] == deg and wind2_deg[i] != wind1_deg[i]:  # 避免同一天同一风向重复计算
                speed.append(wind_speed[i])
        # 计算该方向的平均风速
        if speed:
            avg_speed = sum(speed) / len(speed)
        else:
            avg_speed = 0
        temp.append(avg_speed)
    
    print(temp)
    N = 8
    # 设置theta为风向角度的弧度值
    theta = np.deg2rad(degs)
    radii = np.array(temp)
    
    plt.figure(figsize=(10, 10))  # 设置雷达图大小
    plt.axes(polar=True)
    
    # 设置北方向在顶部
    plt.gca().set_theta_zero_location('N')
    plt.gca().set_theta_direction(-1)  # 顺时针方向
    
    # 使用更直观的颜色映射（蓝色=低风速，红色=高风速）
    if max(radii) > 0:
        from matplotlib.cm import viridis
        colors = viridis(radii / max(radii))
    else:
        colors = [(0.5, 0.5, 0.6)] * N  # 所有风速为零时的默认颜色
    
    # 创建以每个风向角度为中心的条形图
    width = 2 * np.pi / N
    bars = plt.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.8)
    
    # 添加风向标签
    plt.xticks(theta, directions, fontsize=12, fontweight='bold')
    
    # 添加风速网格线以提高可读性
    max_radius = max(radii)
    if max_radius > 0:
        # 设置合适的网格线数量
        grid_steps = 3
        grid_values = np.linspace(0, max_radius, grid_steps + 1)
        for r in grid_values[1:]:  # 跳过0值网格线
            plt.polar(np.linspace(0, 2*np.pi, 100), [r]*100, 'k--', alpha=0.3, linewidth=0.8)
    
    # 添加风速值标签
    for i, (r, t) in enumerate(zip(radii, theta)):
        if r > 0:
            plt.text(t, r + 0.2, f'{r:.1f}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.title('7日平均风力风向雷达图', fontsize=16, fontweight='bold', pad=20)
    
    # 添加颜色条
    if max(radii) > 0:
        from matplotlib.cm import viridis
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=max_radius))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', pad=0.1, aspect=50)
        cbar.set_label('平均风速', fontsize=12, fontweight='bold')
    
    # 保存图片
    plt.savefig(os.path.join(dir_path, 'wind_radar.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_path, 'wind_radar.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    print('雷达图已保存为 wind_radar.png 和 wind_radar.jpg')
    plt.show()
wind_radar(df_week, script_dir)

#绘制7日天气分布饼图
def weather_pie(df, dir_path):
    weather = list(df['weather'])
    weather_count = {}
    for w in weather:
        if w in weather_count:
            weather_count[w] += 1
        else:
            weather_count[w] = 1
    
    # 绘制饼图
    plt.figure(figsize=(8, 8))
    plt.pie(weather_count.values(), labels=weather_count.keys(), autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
    plt.title('7日天气分布饼图', fontsize=16, fontweight='bold', pad=20)
    
    # 保存图片
    plt.savefig(os.path.join(dir_path, 'weather_pie.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_path, 'weather_pie.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    print('饼图已保存为 weather_pie.png 和 weather_pie.jpg')
    plt.show()
weather_pie(df_week, script_dir)

       


