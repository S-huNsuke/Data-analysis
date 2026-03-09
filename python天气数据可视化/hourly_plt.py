# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
import os

# 中文字符设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#读取数据并进行处理
def data_wash():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用绝对路径读取CSV文件
    csv_path = os.path.join(script_dir, 'hourly_weather_data.csv')
    df_hourly = pd.read_csv(csv_path)
    hourly_data = df_hourly[::-1].reset_index(drop=True)
    shunxu = []
    for i in range(25):
        shunxu.append(i)
    hourly_data = hourly_data.assign(shunxu=shunxu)
    return hourly_data, script_dir
#绘制温度变化折线图
def plt_hourly_temp():
    hourly_data, script_dir = data_wash()
    sns.lineplot( x='shunxu', y='temperature', data=hourly_data, label='气温', color='#1f77b4', linewidth=2.5, marker='o', markersize=8)
    average_temp = hourly_data['temperature'].mean()
    avg_line = plt.axhline(average_temp, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('24小时温度趋势图', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('时间', fontsize=12, fontweight='bold')
    plt.ylabel('温度 (℃)', fontsize=12, fontweight='bold')
    plt.text(hourly_data['shunxu'].iloc[0], average_temp, f'平均气温: {average_temp:.1f}℃', 
            ha='left', va='bottom', fontsize=10, color='#1f77b4', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1f77b4', alpha=0.9, linewidth=1.2))
    
    # 获取当前坐标轴
    ax = plt.gca()
    
    # 设置x轴刻度从0开始，增加刻度数量
    plt.xticks(range(0, 25, 2), fontsize=10, rotation=30)
    
    # 设置x轴的0点对齐到原点
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # 设置刻度方向
    ax.tick_params(direction='in', which='both')
    
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
    plt.gca().set_facecolor('#f8f9fa')

    plt.savefig(os.path.join(script_dir, '24weather_trend.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, '24weather_trend.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    print(f'图片已保存为 {os.path.join(script_dir, "24weather_trend.png")} 和 {os.path.join(script_dir, "24weather_trend.jpg")}')
    plt.show()
plt_hourly_temp()
#绘制湿度变化折线图
def plt_hourly_humidity():
    hourly_data, script_dir = data_wash()
    sns.lineplot( x='shunxu', y='humidity', data=hourly_data, label='湿度', color='#1f77b4', linewidth=2.5, marker='o', markersize=8)
    average_humidity = hourly_data['humidity'].mean()
    avg_line = plt.axhline(average_humidity, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('24小时湿度趋势图', fontsize=16, fontweight='bold', pad=20)   
    plt.xlabel('时间', fontsize=12, fontweight='bold')
    plt.ylabel('湿度 (%)', fontsize=12, fontweight='bold')
    plt.text(hourly_data['shunxu'].iloc[0], average_humidity, f'平均湿度: {average_humidity:.1f}%', 
            ha='left', va='bottom', fontsize=10, color='#1f77b4', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1f77b4', alpha=0.9, linewidth=1.2))
    
    # 获取当前坐标轴
    ax = plt.gca()
    
    # 设置x轴刻度从0开始，增加刻度数量
    plt.xticks(range(0, 25, 2), fontsize=10, rotation=30)
    
    # 设置x轴的0点对齐到原点
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # 设置刻度方向
    ax.tick_params(direction='in', which='both')
    
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
    plt.gca().set_facecolor('#f8f9fa')

    plt.savefig(os.path.join(script_dir, '24hourly_hum_trend.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, '24hourly_hum_trend.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    print(f'图片已保存为 {os.path.join(script_dir, "24hourly_hum_trend.png")} 和 {os.path.join(script_dir, "24hourly_hum_trend.jpg")}')
    plt.show()
plt_hourly_humidity()
#绘制24小时风力变化雷达图
def wind_switch(wind):
    if wind == '北风':
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
        return wind
def wind_radar(df, dir_path):
    wind = list(df['wind_direction'])
    wind_speed = list(df['wind_level'])
    
    # 将风向转换为角度
    wind_deg = [wind_switch(w) for w in wind]
    
    # 定义8个风向及其正确角度 (0° = 北风)
    directions = ['北风', '东北风', '东风', '东南风', '南风', '西南风', '西风', '西北风']
    degs = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # 每个方向的正确角度
    temp = []
    
    for deg in degs:
        speed = []
        for i in range(25):  # 检查所有25小时数据 (0-24)
            if wind_deg[i] == deg:
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
    
    # 改进颜色方案，使用更直观的颜色映射（蓝色=低风速，红色=高风速）
    if max(radii) > 0:
        # 使用matplotlib的颜色映射
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
    
    plt.title('24小时平均风力风向雷达图', fontsize=16, fontweight='bold', pad=20)
    
    # 添加颜色条
    if max(radii) > 0:
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=max_radius))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', pad=0.1, aspect=50)
        cbar.set_label('平均风速', fontsize=12, fontweight='bold')
    
    # 保存图片
    plt.savefig(os.path.join(dir_path, 'wind_radar.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_path, 'wind_radar.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    print('雷达图已保存为 wind_radar.png 和 wind_radar.jpg')
    plt.show()
def plt_hourly_wind():
    hourly_data, script_dir = data_wash()
    wind_radar(hourly_data, script_dir)
plt_hourly_wind()
#绘制24小时温湿度相关性
def temp_hum(df, dir_path):
    temp = list(df['temperature'])
    hum = list(df['humidity'])
    plt.figure(figsize=(10, 6))
    plt.scatter(temp, hum, alpha=0.5)
    plt.title('24小时温湿度相关性', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('温度 (℃)', fontsize=12, fontweight='bold')
    plt.ylabel('湿度 (%)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
    plt.gca().set_facecolor('#f8f9fa')
    plt.savefig(os.path.join(dir_path, '24hourly_temp_hum.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_path, '24hourly_temp_hum.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    print('温湿度相关性图已保存为 24hourly_temp_hum.png 和 24hourly_temp_hum.jpg')
    plt.show()
def plt_hourly_temp_hum():
    hourly_data, script_dir = data_wash()
    temp_hum(hourly_data, script_dir)
plt_hourly_temp_hum()
