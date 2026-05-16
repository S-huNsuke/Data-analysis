# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import jieba

# 导入pyecharts相关库（用于树状图可视化）
from pyecharts.charts import TreeMap
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# 导入词云相关库
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

# 获取当前工作目录或脚本所在目录（兼容Jupyter notebook）
try:
    # 尝试获取脚本所在目录（适用于普通Python脚本）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本所在目录: {script_dir}")
except NameError:
    # 如果在Jupyter notebook中运行，使用当前工作目录
    script_dir = os.getcwd()
    print(f"当前工作目录: {script_dir}")

output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 中文字符设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 数据读取及清理
def read_and_clean_data():
    excel_file = os.path.join(script_dir, 'B站新榜_综合指数榜单.xlsx')
    df = pd.read_excel(excel_file)
    df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d')
    
    # 转换投币数列的格式
    def convert_coin_value(value):
        if isinstance(value, str):
            if 'w' in value:
                num = float(value.replace('w', ''))
                return int(num * 10000)
            else:
                return int(float(value))
        return value
    df['投币数'] = df['投币数'].apply(convert_coin_value)
    
    # 转换涨粉数列的格式
    def convert_follow_value(value):
        if isinstance(value, str):
            if 'w' in value:
                num = float(value.replace('w', ''))
                return int(num * 10000)
            else:
                return int(float(value))
        return value
    df['涨粉数'] = df['涨粉数'].apply(convert_follow_value)
    
    # 转换获赞数列的格式
    def convert_like_value(value):
        if isinstance(value, str):
            if 'w' in value:
                num = float(value.replace('w', ''))
                return int(num * 10000)
            else:
                return int(float(value))
        return value
    df['获赞数'] = df['获赞数'].apply(convert_like_value)
    
    # 转换播放数列的格式
    def convert_play_value(value):
        if isinstance(value, str):
            value = value.strip()
            if 'w' in value:
                num = float(value.replace('w', ''))
                return int(num * 10000)
            elif '亿' in value:
                num = float(value.replace('亿', ''))
                return int(num * 100000000)
            else:
                return int(float(value))
        return value
    df['播放数'] = df['播放数'].apply(convert_play_value)
    
    # 转换弹幕数列的格式
    def convert_dm_value(value):
        if isinstance(value, str):
            if 'w' in value:
                num = float(value.replace('w', ''))
                return int(num * 10000)
            else:
                return int(float(value))
        return value
    df['弹幕数'] = df['弹幕数'].apply(convert_dm_value)
    
    # 转换投稿视频数的格式
    def convert_video_value(value):
        if isinstance(value, str):
            if 'w' in value:
                num = float(value.replace('w', ''))
                return int(num * 10000)
            else:
                return int(float(value))
        return value
    df['投稿视频数'] = df['投稿视频数'].apply(convert_video_value)
    
    return df

# 性别分布情况
def plot_gender_distribution(data):
    gender_counts = data['性别'].value_counts()
    gender_com = gender_counts.reset_index()
    gender_com.columns = ['性别', 'count']
    
    plt.figure(figsize=(8, 6))
    
    # 计算百分比并创建带百分比的标签
    total = gender_com['count'].sum()
    gender_com['percentage'] = gender_com['count'] / total * 100
    gender_com['combined_label'] = gender_com.apply(lambda x: f"{x['性别']} ({x['percentage']:.1f}%)", axis=1)
    
    wedges, texts = plt.pie(
        gender_com['count'],
        labels=gender_com['combined_label'],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1, 'width': 0.5},
        textprops={'fontsize': 12}
    )
    
    plt.title('B站UP主性别分布', fontsize=16, pad=20)
    plt.axis('equal')
    plt.legend(title='性别', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

# 获赞数Top10 UP主
def create_likes(data):
    data['获赞数'] = data['获赞数'].astype(int)
    up_likes = data.groupby('up主', observed=True)['获赞数'].sum().reset_index()
    sorted_up_likes = up_likes.sort_values(by='获赞数', ascending=False)
    top10 = sorted_up_likes.head(10)
    plt.figure(figsize=(15, 10))
    bars = plt.barh(top10['up主'], top10['获赞数'], color='#FB7299', height=0.6)
    plt.gca().invert_yaxis()
    plt.title('获赞数Top10 UP主', fontsize=18, pad=20)
    plt.xlabel('获赞数', fontsize=14)
    plt.ylabel('UP主', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 2000, bar.get_y() + bar.get_height()/2, f'{width:,}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    for i, bar in enumerate(bars):
        y_center = bar.get_y() + bar.get_height()/2
        plt.text(bar.get_width()/2, y_center, f'{i+1}', ha='center', va='center',
                fontsize=18, fontweight='bold', color='white')

    plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.1)
    plt.show()

# 获赞数分布情况
def create_likes_distribution(data):
    Bins = [0, 10000, 50000, 100000, 250000, 500000, 1000000, 2000000, float('inf')]
    Labels = ['0-10k', '10k-50k', '50k-1m', '1m-2.5m', '2.5m-5m', '5m-10m', '10m-20m', '20m+']
    len_stage = pd.cut(data['获赞数'], bins=Bins, labels=Labels, include_lowest=True).value_counts().sort_index()
    attr = len_stage.index.tolist()
    v1 = len_stage.values.tolist()
    bar = plt.bar(attr, v1, color='#FB7299', width=0.6)
    plt.title('获赞数分布情况', fontsize=18, pad=20)
    plt.xlabel('获赞数范围', fontsize=14)
    plt.ylabel('UP主数量', fontsize=14)
    # 优化x轴标签对齐和可读性
    plt.xticks(rotation=30, ha='center', fontsize=12)
    plt.gca().tick_params(axis='x', pad=10)
    plt.subplots_adjust(bottom=0.15)
    plt.yticks(fontsize=12)
    
    # 在每个柱状图上方添加具体数值
    for bar in bar:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{height:,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.show()

# 涨粉数Top10 UP主
def create_followers(data):
    data['涨粉数'] = data['涨粉数'].astype(int)
    up_followers = data.groupby('up主', observed=True)['涨粉数'].sum().reset_index()
    sorted_up_followers = up_followers.sort_values(by='涨粉数', ascending=False)
    top10 = sorted_up_followers.head(10)
    plt.figure(figsize=(15, 10))
    bars = plt.barh(top10['up主'], top10['涨粉数'], color='#FB7299', height=0.6)
    plt.gca().invert_yaxis()
    plt.title('涨粉数Top10 UP主', fontsize=18, pad=20)
    plt.xlabel('涨粉数', fontsize=14)
    plt.ylabel('UP主', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, ha='right')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 2000, bar.get_y() + bar.get_height()/2, f'{width:,}', 
                ha='left', va='center', fontsize=10, fontweight='bold')

    for i, bar in enumerate(bars):
        y_center = bar.get_y() + bar.get_height()/2
        plt.text(bar.get_width()/2, y_center, f'{i+1}', ha='center', va='center',
                fontsize=18, fontweight='bold', color='white')

    plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.1)
    plt.show()

# 涨粉数分布情况
def create_followers_distribution(data):
    Bins = [0, 1000, 5000, 10000, 25000, 50000, 100000, 200000, float('inf')]
    Labels = ['0-1k', '1k-5k', '5k-10k', '10k-25k', '25k-50k', '50k-100k', '100k-200k', '200k+']
    len_stage = pd.cut(data['涨粉数'], bins=Bins, labels=Labels, include_lowest=True).value_counts().sort_index()
    attr = len_stage.index.tolist()
    v1 = len_stage.values.tolist()
    bar = plt.bar(attr, v1, color='#FB7299', width=0.6)
    plt.title('涨粉数分布情况', fontsize=18, pad=20)
    plt.xlabel('涨粉数范围', fontsize=14)
    plt.ylabel('UP主数量', fontsize=14)
    plt.xticks(rotation=30, ha='center', fontsize=12)
    plt.gca().tick_params(axis='x', pad=10)
    plt.subplots_adjust(bottom=0.15)
    plt.yticks(fontsize=12)
    
    # 在每个柱状图上方添加具体数值
    for bar in bar:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{height:,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.show()

# 投币数Top10 UP主
def create_coins(data):
    data['投币数'] = data['投币数'].astype(int)
    up_coins = data.groupby('up主', observed=True)['投币数'].sum().reset_index()
    sorted_up_coins = up_coins.sort_values(by='投币数', ascending=False)
    top10 = sorted_up_coins.head(10)
    plt.figure(figsize=(15, 10))
    bars = plt.barh(top10['up主'], top10['投币数'], color='#FB7299', height=0.6)
    plt.gca().invert_yaxis()
    plt.title('投币数Top10 UP主', fontsize=18, pad=20)
    plt.xlabel('投币数', fontsize=14)
    plt.ylabel('UP主', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 2000, bar.get_y() + bar.get_height()/2, f'{width:,}', 
                ha='left', va='center', fontsize=10, fontweight='bold')

    for i, bar in enumerate(bars):
        y_center = bar.get_y() + bar.get_height()/2
        plt.text(bar.get_width()/2, y_center, f'{i+1}', ha='center', va='center',
                fontsize=18, fontweight='bold', color='white')

    plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.1)
    plt.show()

# 投币数分布情况
def create_coins_distribution(data):
    Bins = [0, 1000, 3000, 5000, 10000, 15000, 30000, 50000, float('inf')]
    Labels = ['0-1k', '1k-3k', '3k-5k', '5k-10k', '10k-15k', '15k-30k', '30k-50k', '50k+']
    len_stage = pd.cut(data['投币数'], bins=Bins, labels=Labels, include_lowest=True).value_counts().sort_index()
    attr = len_stage.index.tolist()
    v1 = len_stage.values.tolist()

    bars = plt.bar(attr, v1, color='#FB7299', width=0.6)
    plt.title('投币数分布情况', fontsize=18, pad=20)
    plt.xlabel('投币数范围', fontsize=14)
    plt.ylabel('UP主数量', fontsize=14)
    plt.xticks(rotation=30, ha='center', fontsize=12)
    plt.gca().tick_params(axis='x', pad=10)
    plt.subplots_adjust(bottom=0.15)
    plt.yticks(fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{height:,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.show()

# 各类型获赞数分布情况 - 树状图
def create_type_likes_distribution(df):
    # 按类型分组计算总获赞数
    type_likes = df.groupby('类型', observed=True)['获赞数'].sum().reset_index()
    
    # 转换为树状图所需的数据格式
    treemap_data = []
    for _, row in type_likes.iterrows():
        treemap_data.append({
            "name": row['类型'],
            "value": int(row['获赞数'])  # 确保value为整数
        })
    
    # 创建树状图
    treemap = TreeMap(
        init_opts=opts.InitOpts(
            theme=ThemeType.ROMANTIC,
            width="1000px",
            height="600px"
        )
    )
    
    # 添加数据 - 使用pyecharts 2.0.x的核心参数
    treemap.add(
        series_name="各类型UP主获赞数",
        data=treemap_data,  # pyecharts 2.0.x使用data而非data_pair
        label_opts=opts.LabelOpts(
            position="inside",
            color="white",
            font_size=12
        ),
        itemstyle_opts=opts.ItemStyleOpts(
            border_color="#fff",
            border_width=1
        )
    )
    
    # 设置全局配置
    treemap.set_global_opts(
        title_opts=opts.TitleOpts(
            title="各类型UP主获赞数分布",
            title_textstyle_opts=opts.TextStyleOpts(font_size=18)
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{b}: {c} ({d}%)"
        )
    )
    
    # 渲染为HTML文件 - 使用脚本所在目录确保文件保存在main.py文件夹
    html_path = os.path.join(output_dir, "type_likes_treemap.html")
    
    try:
        # 渲染树状图
        result = treemap.render(html_path)
        print(f"脚本所在目录: {script_dir}")
        print(f"树状图已渲染为文件: {html_path}")
        print(f"渲染结果: {result}")
    except Exception as e:
        print(f"渲染树状图时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 同时返回树状图对象（适用于Jupyter Notebook）
    return treemap

# 各类型投币数分布情况 - 树状图
def create_type_coins_distribution(df):
    # 按类型分组计算总投币数
    type_coins = df.groupby('类型', observed=True)['投币数'].sum().reset_index()
    
    # 转换为树状图所需的数据格式
    treemap_data = []
    for _, row in type_coins.iterrows():
        treemap_data.append({
            "name": row['类型'],
            "value": int(row['投币数'])  # 确保value为整数
        })
    
    # 创建树状图
    treemap = TreeMap(
        init_opts=opts.InitOpts(
            theme=ThemeType.ROMANTIC,
            width="1000px",
            height="600px"
        )
    )
    
    # 添加数据 - 使用pyecharts 2.0.x的核心参数
    treemap.add(
        series_name="各类型UP主投币数",
        data=treemap_data,  # pyecharts 2.0.x使用data而非data_pair
        label_opts=opts.LabelOpts(
            position="inside",
            color="white",
            font_size=12
        ),
        itemstyle_opts=opts.ItemStyleOpts(
            border_color="#fff",
            border_width=1
        )
    )
    
    # 设置全局配置
    treemap.set_global_opts(
        title_opts=opts.TitleOpts(
            title="各类型UP主投币数分布",
            title_textstyle_opts=opts.TextStyleOpts(font_size=18)
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{b}: {c} ({d}%)"
        )
    )
    
    # 渲染为HTML文件 - 使用脚本所在目录确保文件保存在main.py文件夹
    html_path = os.path.join(output_dir, "type_coins_treemap.html")
    
    try:
        # 渲染树状图
        result = treemap.render(html_path)
        print(f"脚本所在目录: {script_dir}")
        print(f"树状图已渲染为文件: {html_path}")
        print(f"渲染结果: {result}")
    except Exception as e:
        print(f"渲染树状图时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 同时返回树状图对象（适用于Jupyter Notebook）
    return treemap

# 各类型UP主视频数分布情况 - 树状图
def create_type_video_count_distribution(df):
    # 按类型分组计算视频数
    type_video_count = df.groupby('类型', observed=True).size().reset_index(name='视频数')
    
    # 转换为树状图所需的数据格式
    treemap_data = []
    for _, row in type_video_count.iterrows():
        treemap_data.append({
            "name": row['类型'],
            "value": int(row['视频数'])  # 确保value为整数
        })
    
    # 创建树状图
    treemap = TreeMap(
        init_opts=opts.InitOpts(
            theme=ThemeType.ROMANTIC,
            width="1000px",
            height="600px"
        )
    )
    
    # 添加数据 - 使用pyecharts 2.0.x的核心参数
    treemap.add(
        series_name="各类型UP主视频数",
        data=treemap_data,  # pyecharts 2.0.x使用data而非data_pair
        label_opts=opts.LabelOpts(
            position="inside",
            color="white",
            font_size=12
        ),
        itemstyle_opts=opts.ItemStyleOpts(
            border_color="#fff",
            border_width=1
        )
    )
    
    # 设置全局配置
    treemap.set_global_opts(
        title_opts=opts.TitleOpts(
            title="各类型UP主视频数分布",
            title_textstyle_opts=opts.TextStyleOpts(font_size=18)
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{b}: {c} ({d}%)"
        )
    )
    
    # 渲染为HTML文件 - 使用脚本所在目录确保文件保存在main.py文件夹
    html_path = os.path.join(output_dir, "type_video_count_treemap.html")
    
    try:
        # 渲染树状图
        result = treemap.render(html_path)
        print(f"脚本所在目录: {script_dir}")
        print(f"树状图已渲染为文件: {html_path}")
        print(f"渲染结果: {result}")
    except Exception as e:
        print(f"渲染树状图时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 同时返回树状图对象（适用于Jupyter Notebook）
    return treemap

#UP主标签词云
def create_tag_wordcloud(df):
    stopwords_file = os.path.join(script_dir, 'cn_stopwords.txt')
    words = pd.read_csv(stopwords_file, header=None, encoding='utf-8', names=['stopword'])
    text = ''
    df1 = df[df['up主标签'] != '']
    df2 = df1.copy()
    for tag in df1['up主标签']:
        tag_str = str(tag).replace(' ','') if pd.notna(tag) else ''
        text += tag_str + ' '.join(jieba.cut(tag_str, cut_all=False))
    stopwords = set('')
    stopwords.update(words['stopword'].tolist())
    background_image_file = os.path.join(script_dir, 'bilibili.jpg')
    background_image = plt.imread(background_image_file)
    alice_coloring = np.array(Image.open(background_image_file))
    img_color = ImageColorGenerator(alice_coloring)
    wc = WordCloud(
        background_color='white',
        max_words=2000,
        mask=background_image,
        max_font_size=70,
        min_font_size=1,
        prefer_horizontal=1,
        color_func=img_color,
        width=1000,
        height=600,
        random_state=50,
        stopwords=stopwords,
        margin=5,
        font_path='/System/Library/Fonts/STHeiti Medium.ttc'
    )
    wc.generate_from_text(text)
    process_word = WordCloud.process_text(wc, text)
    sort = sorted(process_word.items(), key=lambda x: x[1], reverse=True)
    print(sort[:50])
    plt.imshow(wc)
    plt.axis('off')
    wc.to_file(os.path.join(output_dir, 'tag_wordcloud.png'))
        
# 主程序入口
if __name__ == "__main__":
    # 读取并清理数据
    df = read_and_clean_data()
    
    # 调用各个可视化函数
    plot_gender_distribution(df)
    create_likes(df)
    create_likes_distribution(df)
    create_followers(df)
    create_followers_distribution(df)
    create_coins(df)
    create_coins_distribution(df)
    
    # 生成树状图
    create_type_likes_distribution(df)
    create_type_coins_distribution(df)
    create_type_video_count_distribution(df)
    
    # 生成UP主标签词云
    create_tag_wordcloud(df)