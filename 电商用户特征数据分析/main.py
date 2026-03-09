# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 中文字符设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 数据读取及处理
def data_wash():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'user_personalized_features.csv')
    df = pd.read_csv(csv_path)
    pd.set_option('display.max_columns', None)
    df['User_ID'] = df['User_ID'].astype('category')
    df['Age'] = df['Age'].astype('category')
    df['Gender'] = df['Gender'].astype('category')
    df['Location'] = df['Location'].astype('category')
    df['Interests'] = df['Interests'].astype('category')
    df['Product_Category_Preference'] = df['Product_Category_Preference'].astype('category')

    #修改列名
    column_map = {
    'User_ID': '用户ID',
    'Age': '年龄',
    'Gender': '性别',
    'Location': '位置',
    'Income': '收入',
    'Interests': '兴趣',
    'Last_Login_Days_Ago': '最近登录时间',
    'Purchase_Frequency': '购买频率',
    'Average_Order_Value': '平均消费金额',
    'Total_Spending': '总消费金额',
    'Product_Category_Preference': '产品类别偏好',
    'Time_Spent_on_Site_Minutes': '平台使用时间',
    'Pages_Viewed': '访问页面数',
    'Newsletter_Subscription': '是否订阅活动通知'
    }
    df.rename(columns=column_map, inplace=True)
    gender_map = {
    'Male': '男',
    'Female': '女'
    }
    region_map = {
    'Suburban': '郊区',
    'Rural': '农村',
    'Urban': '城市'
    }
    Product_Category_Preference_map = {
    'Electronics': '电子产品',
    'Home & Kitchen': '家居厨具',
    'Health & Beauty': '健康美容',
    'Sports': '运动',
    'Books': '图书',
    'Apparel': '服装'
    }
    Interests_map = {
    'Sports': '运动',
    'Technology': '科技',
    'Fashion': '时尚',
    'Travel': '旅行',
    'Food': '美食'
    }

    # 映射性别、位置、产品类别偏好、兴趣
    df['性别'] = df['性别'].map(gender_map)
    df['位置'] = df['位置'].map(region_map)
    df['产品类别偏好'] = df['产品类别偏好'].map(Product_Category_Preference_map)
    df['兴趣'] = df['兴趣'].map(Interests_map)

    df = df.drop(columns=['Unnamed: 0'])

    return df
data = data_wash()

#增加年龄群体和收入群体
def add_label(df):
    def age_label(age):
        if age <= 17:
            return '少年'
        elif age <= 35:
            return '青年'
        elif age <= 45:
            return '中年'
        elif age <= 55:
            return '中老年'
        else:
            return '老年'

    quantiles = df['收入'].quantile([0.25, 0.5, 0.75])
    def income_label(income, quantiles):
        if income <= quantiles[0.25]:
            return '低收入'
        elif income <= quantiles[0.5]:
            return '中低收入'
        elif income <= quantiles[0.75]:
            return '中高收入'
        else:
            return '高收入'

    
    df['年龄群体'] = df['年龄'].apply(age_label)
    df['收入群体'] = df['收入'].apply(lambda x: income_label(x, quantiles))

    df['R_Score'] = pd.qcut(df['最近登录时间'], q=5, labels=[5, 4, 3, 2, 1])
    df['M_Score'] = pd.qcut(df['总消费金额'], q=5, labels=[1, 2, 3, 4, 5])    
    df['F_Score'] = pd.qcut(df['购买频率'], q=5, labels=[1, 2, 3, 4, 5])
    df['RFM_Score'] = df['R_Score'].astype(str) + df['M_Score'].astype(str) + df['F_Score'].astype(str)

    def RFM_label(R_Score, M_Score, F_Score):
        if R_Score >= 4 and M_Score >= 4 and F_Score >= 4:
            return '高价值客户'
        elif R_Score >= 4 and F_Score <= 2 and M_Score <= 2:
            return '新客户'
        elif R_Score <= 2 and F_Score >= 4 and M_Score >= 4:  
            return '易流失高价值客户'
        elif R_Score <= 2 and F_Score <= 2 and M_Score <= 2:
            return '流失用户'
        elif R_Score >= 4:
            return '活跃用户'
        elif F_Score >= 4:
            return '高购买频率用户'
        elif M_Score >= 4:
            return '高消费用户'
        else:
            return '其他客户'

    df['RFM_Group'] = df[['R_Score', 'M_Score', 'F_Score']].apply(lambda x: RFM_label(*x), axis=1)

    df['性别'] = df['性别'].astype(str)
    df['位置'] = df['位置'].astype(str)
    df['产品类别偏好'] = df['产品类别偏好'].astype(str)
    df['兴趣'] = df['兴趣'].astype(str)
    df['年龄群体'] = df['年龄群体'].astype(str)

    return df
add_label(data)


#绘制性别分布饼图及年龄群体性别分布柱状图
def plot_gender_age_distribution(data):

    #按年龄群体分组
    age_grouped = data.groupby('年龄群体').agg({
        '性别': lambda x: x.value_counts().to_dict(),
        '位置': lambda x: x.value_counts().to_dict(),
        '产品类别偏好': lambda x: x.value_counts().to_dict(),
        '兴趣': lambda x: x.value_counts().to_dict(),
        '用户ID': 'count'
    }).rename(columns={'用户ID': '用户数量'})
    print('按年龄群体分组统计结果:')
    for group, stats in age_grouped.iterrows():
        print(f"\n{group}用户统计:")
        print(f"性别分布: {stats['性别']}")
        print(f"位置分布: {stats['位置']}")
        print(f"产品类别偏好分布: {stats['产品类别偏好']}")
        print(f"兴趣分布: {stats['兴趣']}")
        print(f"用户数量: {stats['用户数量']}")

    #绘制饼图
    gender_counts = data['性别'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('性别分布')
    plt.axis('equal')
    plt.show()

    #绘制年龄群体性别分布柱状图
    age_gender = pd.crosstab(data['年龄群体'], data['性别'])
    age_gender.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title('年龄群体性别分布')
    plt.xlabel('年龄群体')
    plt.ylabel('用户数量')
    plt.xticks(rotation=0)
    plt.show()
plot_gender_age_distribution(data)

#按收入群体统计
def plot_income_age_distribution(data):
    #按收入群体分组
    income_grouped = data.groupby('收入群体').agg({
        '性别': lambda x: x.value_counts().to_dict(),
        '位置': lambda x: x.value_counts().to_dict(),
        '兴趣': lambda x: x.value_counts().to_dict(),
        '用户ID': 'count'
    }).rename(columns={'用户ID': '用户数量'})
    print('按收入群体分组统计结果:')
    for group, stats in income_grouped.iterrows():
        print(f"\n{group}用户统计:")
        print(f"性别分布: {stats['性别']}")
        print(f"位置分布: {stats['位置']}")
        print(f"兴趣分布: {stats['兴趣']}")
        print(f"用户数量: {stats['用户数量']}")
plot_income_age_distribution(data)

#按地区分布柱状图
def plot_location_distribution(data):
    plt.figure(figsize=(10, 6))
    location_counts = data['位置'].value_counts()
    location_counts.plot(kind='bar', color='skyblue')
    plt.title('位置分布')
    plt.xlabel('位置')
    plt.ylabel('用户数量')
    plt.xticks(rotation=0)  # 将x轴文字改为水平
    for i in range(len(location_counts)):
        plt.text(i, location_counts.iloc[i], str(location_counts.iloc[i]), ha='center', va='bottom')
    plt.show()
plot_location_distribution(data)

#按兴趣分布饼图
def plot_interest_distribution(data):
    plt.figure(figsize=(8, 8))
    interest_counts = data['兴趣'].value_counts()
    wedges, texts, autotexts = plt.pie(
        interest_counts.values,
        labels=interest_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )
    plt.title('兴趣分布', fontsize=14)
    plt.axis('equal')  # 确保饼图是正圆形
    plt.tight_layout()
    plt.show()
plot_interest_distribution(data)

#按照订阅活动统计
def plot_subscription_distribution(data):
    #按订阅活动分组
    subscription_grouped = data.groupby('是否订阅活动通知').agg({
        '平台使用时间': lambda x: x.sum(),
        '访问页面数': lambda x: x.sum(),
        '购买频率': lambda x: x.sum(),
        '总消费金额': lambda x: x.sum(),
        '用户ID': 'count'
    }).rename(columns={'用户ID': '用户数量'})
    print('按订阅活动分组统计结果:')
    for group, stats in subscription_grouped.iterrows():
        print(f"\n{group}用户统计:")
        print(f"平台使用时间: {stats['平台使用时间']}")
        print(f"访问页面数分布: {stats['访问页面数']}")
        print(f"购买频率分布: {stats['购买频率']}")
        print(f"总消费金额: {stats['总消费金额']}")
        print(f"用户数量: {stats['用户数量']}")
plot_subscription_distribution(data)

#计算整体客单价
def calculate_average_purchase_price(data):
    total_revenue = data['总消费金额'].sum()
    total_purchases = data['用户ID'].nunique()
    if total_purchases > 0:
        average_purchase_price = total_revenue / total_purchases
        print(f"总消费金额: {total_revenue:.2f}")
        print(f"总购买次数: {total_purchases}")
        print(f"整体客单价: {average_purchase_price:.2f}")
    else:
        average_purchase_price = 0
        print("总购买次数为0，无法计算客单价")

    #基于是否订阅的客单价
    subscription_grouped = data.groupby('是否订阅活动通知').agg({
        '总消费金额': 'sum',
        '用户ID': 'nunique'
    })
    subscription_grouped['客单价'] = subscription_grouped['总消费金额'] / subscription_grouped['用户ID']
    print("\n基于是否订阅的客单价:")
    for group, stats in subscription_grouped.iterrows():
        print(f"{group}用户客单价: {stats['客单价']:.2f}")
calculate_average_purchase_price(data)