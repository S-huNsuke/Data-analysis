#!/usr/bin/env python3
"""
泰坦尼克号乘客生存预测 - 数据挖掘流程实战
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
import xgboost as xg

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


def load_data(file_path='train.csv'):
    data = pd.read_csv(file_path)
    return data


def explore_data(data):
    print("=" * 60)
    print("数据基本信息")
    print("=" * 60)
    print(f"数据形状: {data.shape}")
    print("\n前5行数据:")
    print(data.head())
    print("\n缺失值统计:")
    print(data.isnull().sum())
    print("\n数据描述性统计:")
    print(data.describe())


def visualize_survival_rate(data):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Survived'].value_counts().plot.pie(
        explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True
    )
    ax[0].set_title('Survived')
    ax[0].set_ylabel('')
    sns.countplot('Survived', data=data, ax=ax[1])
    ax[1].set_title('Survived')
    plt.tight_layout()
    plt.savefig('survival_rate.png', dpi=150)
    plt.show()


def analyze_by_gender(data):
    print("\n性别与生存率交叉表:")
    print(data.groupby(['Sex', 'Survived'])['Survived'].count())

    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
    ax[0].set_title('Survived vs Sex')
    sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])
    ax[1].set_title('Sex: Survived vs Dead')
    plt.tight_layout()
    plt.savefig('survival_by_gender.png', dpi=150)
    plt.show()


def analyze_by_pclass(data):
    print("\n船舱等级与生存率交叉表:")
    print(pd.crosstab(data.Pclass, data.Survived, margins=True))

    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Pclass'].value_counts().plot.bar(
        color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0]
    )
    ax[0].set_title('Number Of Passengers By Pclass')
    ax[0].set_ylabel('Count')
    sns.countplot('Pclass', hue='Survived', data=data, ax=ax[1])
    ax[1].set_title('Pclass: Survived vs Dead')
    plt.tight_layout()
    plt.savefig('survival_by_pclass.png', dpi=150)
    plt.show()


def analyze_pclass_gender(data):
    print("\n船舱等级和性别交叉表:")
    print(pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True))

    sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)
    plt.savefig('survival_pclass_gender.png', dpi=150)
    plt.show()


def analyze_age(data):
    print("\n年龄统计:")
    print(f"最大年龄: {data['Age'].max()} 岁")
    print(f"最小年龄: {data['Age'].min()} 岁")
    print(f"平均年龄: {data['Age'].mean():.2f} 岁")

    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.violinplot("Pclass", "Age", hue="Survived", data=data, split=True, ax=ax[0])
    ax[0].set_title('Pclass and Age vs Survived')
    ax[0].set_yticks(range(0, 110, 10))
    sns.violinplot("Sex", "Age", hue="Survived", data=data, split=True, ax=ax[1])
    ax[1].set_title('Sex and Age vs Survived')
    ax[1].set_yticks(range(0, 110, 10))
    plt.tight_layout()
    plt.savefig('survival_by_age.png', dpi=150)
    plt.show()


def extract_titles(data):
    # 从姓名中提取称谓
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')
    print("\n原始称谓统计:")
    print(pd.crosstab(data.Initial, data.Sex).T)

    # 标准化称谓
    data['Initial'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countness',
         'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs',
         'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'],
        inplace=True
    )
    return data


def fill_missing_age(data):
    # 用各称谓组均值填充年龄缺失值
    print("\n各称谓组平均年龄:")
    print(data.groupby('Initial')['Age'].mean())

    age_means = data.groupby('Initial')['Age'].mean()
    for initial, mean_age in age_means.items():
        data.loc[(data.Age.isnull()) & (data.Initial == initial), 'Age'] = mean_age

    print(f"\n年龄缺失值填充后检查: {data.Age.isnull().any()}")
    return data


def analyze_embarked(data):
    print("\n登船港口交叉表:")
    print(pd.crosstab([data.Embarked, data.Pclass], [data.Sex, data.Survived], margins=True))

    sns.factorplot('Embarked', 'Survived', data=data)
    plt.savefig('survival_by_embarked.png', dpi=150)
    plt.show()

    f, ax = plt.subplots(2, 2, figsize=(20, 15))
    sns.countplot('Embarked', data=data, ax=ax[0, 0])
    ax[0, 0].set_title('No. Of Passengers Boarded')
    sns.countplot('Embarked', hue='Sex', data=data, ax=ax[0, 1])
    ax[0, 1].set_title('Male-Female Split for Embarked')
    sns.countplot('Embarked', hue='Survived', data=data, ax=ax[1, 0])
    ax[1, 0].set_title('Embarked vs Survived')
    sns.countplot('Embarked', hue='Pclass', data=data, ax=ax[1, 1])
    ax[1, 1].set_title('Embarked vs Pclass')
    plt.tight_layout()
    plt.savefig('embarked_analysis.png', dpi=150)
    plt.show()

    sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=data)
    plt.savefig('survival_pclass_gender_embarked.png', dpi=150)
    plt.show()


def analyze_family_size(data):
    print("\n兄弟姐妹/配偶数量与生存率交叉表:")
    print(pd.crosstab([data.SibSp], data.Survived))

    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.barplot('SibSp', 'Survived', data=data, ax=ax[0])
    ax[0].set_title('SibSp vs Survived')
    sns.factorplot('SibSp', 'Survived', data=data, ax=ax[1])
    ax[1].set_title('SibSp vs Survived')
    plt.close(2)
    plt.tight_layout()
    plt.savefig('survival_by_sibsp.png', dpi=150)
    plt.show()

    print("\n父母/孩子数量与船舱等级交叉表:")
    print(pd.crosstab(data.Parch, data.Pclass))

    # 创建家庭规模和是否独行特征
    data['Family_Size'] = data['Parch'] + data['SibSp']
    data['Alone'] = 0
    data.loc[data.Family_Size == 0, 'Alone'] = 1

    f, ax = plt.subplots(1, 2, figsize=(18, 6))
    sns.factorplot('Family_Size', 'Survived', data=data, ax=ax[0])
    ax[0].set_title('Family_Size vs Survived')
    sns.factorplot('Alone', 'Survived', data=data, ax=ax[1])
    ax[1].set_title('Alone vs Survived')
    plt.close(2)
    plt.close(3)
    plt.tight_layout()
    plt.savefig('survival_family_alone.png', dpi=150)
    plt.show()

    sns.factorplot('Alone', 'Survived', data=data, hue='Sex', col='Pclass')
    plt.savefig('survival_alone_gender_pclass.png', dpi=150)
    plt.show()


def create_age_bands(data):
    # 将年龄分为5个区间
    data['Age_band'] = 0
    data.loc[data['Age'] <= 16, 'Age_band'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_band'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age_band'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age_band'] = 3
    data.loc[data['Age'] > 64, 'Age_band'] = 4

    print("\n年龄段分布:")
    print(data['Age_band'].value_counts())

    sns.factorplot('Age_band', 'Survived', data=data, col='Pclass')
    plt.savefig('survival_age_band_pclass.png', dpi=150)
    plt.show()

    return data


def create_fare_categories(data):
    # 将票价离散化为4个类别
    print(f"\n票价统计: 最高={data['Fare'].max()}, 最低={data['Fare'].min()}, 平均={data['Fare'].mean():.2f}")

    f, ax = plt.subplots(1, 3, figsize=(20, 8))
    for i, pclass in enumerate([1, 2, 3]):
        sns.distplot(data[data['Pclass'] == i + 1].Fare, ax=ax[i])
        ax[i].set_title(f'Fares in Pclass {i + 1}')
    plt.tight_layout()
    plt.savefig('fare_distribution.png', dpi=150)
    plt.show()

    data['Fare_cat'] = 0
    data.loc[data['Fare'] <= 7.91, 'Fare_cat'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat'] = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513), 'Fare_cat'] = 3

    sns.factorplot('Fare_cat', 'Survived', data=data, hue='Sex')
    plt.savefig('survival_fare_category.png', dpi=150)
    plt.show()

    return data


def encode_categorical_features(data):
    # 分类特征数值编码
    data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    data['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace=True)
    return data


def visualize_correlation(data):
    sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
    plt.savefig('correlation_matrix.png', dpi=150)
    plt.show()


def prepare_features(data):
    # 删除不需要的列
    data.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_Range', 'PassengerId'],
              axis=1, inplace=True)
    return data


def split_data(data, test_size=0.3, random_state=0):
    train, test = train_test_split(data, test_size=test_size, random_state=random_state,
                                   stratify=data['Survived'])
    train_X = train[train.columns[1:]]
    train_Y = train[train.columns[:1]]
    test_X = test[test.columns[1:]]
    test_Y = test[test.columns[:1]]
    X = data[data.columns[1:]]
    Y = data['Survived']
    return train_X, train_Y, test_X, test_Y, X, Y


def train_single_model(model, train_X, train_Y, test_X, test_Y, model_name):
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = accuracy_score(prediction, test_Y)
    print(f"{model_name} 测试集准确率: {accuracy:.4f}")
    return accuracy


def evaluate_knn_different_k(train_X, train_Y, test_X, test_Y):
    # 测试不同K值对KNN的影响
    a_index = list(range(1, 11))
    a = pd.Series()
    x = list(range(11))

    for i in range(1, 11):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(train_X, train_Y)
        prediction = model.predict(test_X)
        a = a.append(pd.Series(accuracy_score(prediction, test_Y)))

    plt.plot(a_index, a)
    plt.xticks(x)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.savefig('knn_k_selection.png', dpi=150)
    plt.show()

    print(f'不同K值的准确率: {a.values}, 最大值: {a.values.max()}')


def cross_validate_models(X, Y):
    # 10折交叉验证评估多个模型
    kfold = KFold(n_splits=10, random_state=22)
    xyz = []
    accuracy = []
    std = []
    classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN',
                   'Decision Tree', 'Naive Bayes', 'Random Forest']
    models = [
        svm.SVC(kernel='linear'),
        svm.SVC(kernel='rbf'),
        LogisticRegression(),
        KNeighborsClassifier(n_neighbors=9),
        DecisionTreeClassifier(),
        GaussianNB(),
        RandomForestClassifier(n_estimators=100)
    ]

    for i, model in enumerate(models):
        cv_result = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
        xyz.append(cv_result.mean())
        std.append(cv_result.std())
        accuracy.append(cv_result)

    new_models_dataframe2 = pd.DataFrame({'CV Mean': xyz, 'Std': std}, index=classifiers)
    print("\n各模型交叉验证结果:")
    print(new_models_dataframe2)

    plt.subplots(figsize=(12, 6))
    box = pd.DataFrame(accuracy, index=[classifiers])
    box.T.boxplot()
    plt.savefig('model_comparison_boxplot.png', dpi=150)
    plt.show()

    new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
    plt.title('Average CV Mean Accuracy')
    plt.savefig('model_comparison_bar.png', dpi=150)
    plt.show()

    return classifiers, accuracy


def plot_confusion_matrices(X, Y):
    # 绘制各模型混淆矩阵
    f, ax = plt.subplots(3, 3, figsize=(12, 10))

    models = [
        ('rbf-SVM', svm.SVC(kernel='rbf')),
        ('Linear-SVM', svm.SVC(kernel='linear')),
        ('KNN', KNeighborsClassifier(n_neighbors=9)),
        ('Random-Forests', RandomForestClassifier(n_estimators=100)),
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Naive Bayes', GaussianNB())
    ]

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]

    for (name, model), pos in zip(models, positions):
        y_pred = cross_val_predict(model, X, Y, cv=10)
        sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[pos[0], pos[1]],
                   annot=True, fmt='2.0f')
        ax[pos[0], pos[1]].set_title(f'Matrix for {name}')

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig('confusion_matrices.png', dpi=150)
    plt.show()


def grid_search_svm(X, Y):
    # SVM超参数网格搜索
    C = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kernel = ['rbf', 'linear']
    hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}

    gd = GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
    gd.fit(X, Y)
    print(f"SVM 最佳分数: {gd.best_score_}")
    print(f"SVM 最佳参数: {gd.best_estimator_}")


def grid_search_random_forest(X, Y):
    # 随机森林超参数网格搜索
    n_estimators = range(100, 1000, 100)
    hyper = {'n_estimators': n_estimators}

    gd = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                     param_grid=hyper, verbose=True)
    gd.fit(X, Y)
    print(f"随机森林 最佳分数: {gd.best_score_}")
    print(f"随机森林 最佳参数: {gd.best_estimator_}")


def ensemble_voting(train_X, train_Y, test_X, test_Y, X, Y):
    # 投票集成多个模型
    ensemble_lin_rbf = VotingClassifier(
        estimators=[
            ('KNN', KNeighborsClassifier(n_neighbors=10)),
            ('RBF', svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.1)),
            ('RFor', RandomForestClassifier(n_estimators=500, random_state=0)),
            ('LR', LogisticRegression(C=0.05)),
            ('DT', DecisionTreeClassifier(random_state=0)),
            ('NB', GaussianNB()),
            ('svm', svm.SVC(kernel='linear', probability=True))
        ],
        voting='soft'
    ).fit(train_X, train_Y)

    print(f"投票集成模型 测试集准确率: {ensemble_lin_rbf.score(test_X, test_Y):.4f}")
    cross = cross_val_score(ensemble_lin_rbf, X, Y, cv=10, scoring="accuracy")
    print(f"投票集成模型 交叉验证分数: {cross.mean():.4f}")


def ensemble_bagging_knn(train_X, train_Y, test_X, test_Y, X, Y):
    # Bagging集成KNN
    model = BaggingClassifier(
        base_estimator=KNeighborsClassifier(n_neighbors=3),
        random_state=0, n_estimators=700
    )
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    print(f"Bagging KNN 测试集准确率: {accuracy_score(prediction, test_Y):.4f}")
    result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
    print(f"Bagging KNN 交叉验证分数: {result.mean():.4f}")


def ensemble_bagging_tree(train_X, train_Y, test_X, test_Y, X, Y):
    # Bagging集成决策树
    model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        random_state=0, n_estimators=100
    )
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    print(f"Bagging Decision Tree 测试集准确率: {accuracy_score(prediction, test_Y):.4f}")
    result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
    print(f"Bagging Decision Tree 交叉验证分数: {result.mean():.4f}")


def ensemble_adaboost(X, Y):
    ada = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.1)
    result = cross_val_score(ada, X, Y, cv=10, scoring='accuracy')
    print(f"AdaBoost 交叉验证分数: {result.mean():.4f}")


def ensemble_gradient_boosting(X, Y):
    grad = GradientBoostingClassifier(n_estimators=500, random_state=0, learning_rate=0.1)
    result = cross_val_score(grad, X, Y, cv=10, scoring='accuracy')
    print(f"Gradient Boosting 交叉验证分数: {result.mean():.4f}")


def adaboost_grid_search(X, Y):
    # AdaBoost超参数网格搜索
    n_estimators = list(range(100, 1101, 100))
    learn_rate = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    hyper = {'n_estimators': n_estimators, 'learning_rate': learn_rate}

    gd = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=hyper, verbose=True)
    gd.fit(X, Y)
    print(f"AdaBoost 最佳分数: {gd.best_score_}")
    print(f"AdaBoost 最佳参数: {gd.best_estimator_}")


def plot_adaboost_confusion_matrix(X, Y):
    ada = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.05)
    result = cross_val_predict(ada, X, Y, cv=10)
    sns.heatmap(confusion_matrix(Y, result), cmap='winter', annot=True, fmt='2.0f')
    plt.savefig('adaboost_confusion_matrix.png', dpi=150)
    plt.show()


def plot_feature_importance(X, Y):
    # 各模型特征重要性可视化
    f, ax = plt.subplots(2, 2, figsize=(15, 12))

    model = RandomForestClassifier(n_estimators=500, random_state=0)
    model.fit(X, Y)
    pd.Series(model.feature_importances_, X.columns).sort_values(
        ascending=True
    ).plot.barh(width=0.8, ax=ax[0, 0], color='steelblue')
    ax[0, 0].set_title('Feature Importance in Random Forests')

    model = AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=0)
    model.fit(X, Y)
    pd.Series(model.feature_importances_, X.columns).sort_values(
        ascending=True
    ).plot.barh(width=0.8, ax=ax[0, 1], color='#ddff11')
    ax[0, 1].set_title('Feature Importance in AdaBoost')

    model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=0)
    model.fit(X, Y)
    pd.Series(model.feature_importances_, X.columns).sort_values(
        ascending=True
    ).plot.barh(width=0.8, ax=ax[1, 0], cmap='RdYlGn_r')
    ax[1, 0].set_title('Feature Importance in Gradient Boosting')

    model = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
    model.fit(X, Y)
    pd.Series(model.feature_importances_, X.columns).sort_values(
        ascending=True
    ).plot.barh(width=0.8, ax=ax[1, 1], color='#FD0F00')
    ax[1, 1].set_title('Feature Importance in XgBoost')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()


def main():
    print("=" * 60)
    print("泰坦尼克号乘客生存预测 - 数据挖掘流程")
    print("=" * 60)

    # 加载数据
    data = load_data('train.csv')
    explore_data(data)

    # 探索性数据分析
    visualize_survival_rate(data)
    analyze_by_gender(data)
    analyze_by_pclass(data)
    analyze_pclass_gender(data)
    analyze_age(data)

    # 特征工程
    data = extract_titles(data)
    data = fill_missing_age(data)
    analyze_embarked(data)
    analyze_family_size(data)
    data = create_age_bands(data)
    data = create_fare_categories(data)
    data = encode_categorical_features(data)

    # 填充登船港口缺失值
    data['Embarked'].fillna('S', inplace=True)

    # 准备特征
    visualize_correlation(data)
    data = prepare_features(data)

    # 划分数据集
    train_X, train_Y, test_X, test_Y, X, Y = split_data(data)

    # 训练单个模型
    print("\n" + "=" * 60)
    print("模型训练与评估")
    print("=" * 60)

    models = [
        (svm.SVC(kernel='rbf', C=1, gamma=0.1), "RBF SVM"),
        (svm.SVC(kernel='linear', C=0.1, gamma=0.1), "Linear SVM"),
        (LogisticRegression(), "Logistic Regression"),
        (DecisionTreeClassifier(), "Decision Tree"),
        (KNeighborsClassifier(), "KNN"),
        (GaussianNB(), "Naive Bayes"),
        (RandomForestClassifier(n_estimators=100), "Random Forest")
    ]

    for model, name in models:
        train_single_model(model, train_X, train_Y, test_X, test_Y, name)

    # KNN K值选择
    evaluate_knn_different_k(train_X, train_Y, test_X, test_Y)

    # 交叉验证评估
    classifiers, accuracy = cross_validate_models(X, Y)

    # 混淆矩阵
    plot_confusion_matrices(X, Y)

    # 网格搜索优化
    print("\n" + "=" * 60)
    print("超参数优化")
    print("=" * 60)
    grid_search_svm(X, Y)
    grid_search_random_forest(X, Y)

    # 集成学习
    print("\n" + "=" * 60)
    print("模型集成")
    print("=" * 60)
    ensemble_voting(train_X, train_Y, test_X, test_Y, X, Y)
    ensemble_bagging_knn(train_X, train_Y, test_X, test_Y, X, Y)
    ensemble_bagging_tree(train_X, train_Y, test_X, test_Y, X, Y)
    ensemble_adaboost(X, Y)
    ensemble_gradient_boosting(X, Y)

    # AdaBoost网格搜索
    adaboost_grid_search(X, Y)

    # 最佳AdaBoost模型混淆矩阵
    plot_adaboost_confusion_matrix(X, Y)

    # 特征重要性
    plot_feature_importance(X, Y)

    print("\n" + "=" * 60)
    print("分析完成！所有图表已保存。")
    print("=" * 60)


if __name__ == "__main__":
    main()