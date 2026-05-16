"""
机器学习建模模块
提供常用的机器学习模型训练和评估功能
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    silhouette_score,
    r2_score
)
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    训练逻辑回归模型
    
    参数:
        X: 特征数据，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        test_size: 测试集比例，默认为 0.2
        random_state: 随机种子，默认为 42
    
    返回:
        包含以下键的字典:
            - model: 训练好的模型
            - accuracy: 准确率
            - report: 分类报告
            - feature_importance: 特征重要性（使用系数绝对值）
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    feature_importance = np.abs(model.coef_).mean(axis=0)
    
    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "feature_importance": feature_importance
    }


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    训练随机森林分类模型
    
    参数:
        X: 特征数据，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        n_estimators: 决策树数量，默认为 100
        test_size: 测试集比例，默认为 0.2
        random_state: 随机种子，默认为 42
    
    返回:
        包含以下键的字典:
            - model: 训练好的模型
            - accuracy: 准确率
            - report: 分类报告
            - feature_importance: 特征重要性
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    feature_importance = model.feature_importances_
    
    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "feature_importance": feature_importance
    }


def train_kmeans(
    X: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    训练 K-Means 聚类模型
    
    参数:
        X: 特征数据，形状为 (n_samples, n_features)
        n_clusters: 聚类数量，默认为 4
        random_state: 随机种子，默认为 42
    
    返回:
        包含以下键的字典:
            - model: 训练好的模型
            - labels: 聚类标签
            - silhouette_score: 轮廓系数
            - centers: 聚类中心
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, labels)
    centers = model.cluster_centers_
    
    return {
        "model": model,
        "labels": labels,
        "silhouette_score": sil_score,
        "centers": centers
    }


def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    训练线性回归模型
    
    参数:
        X: 特征数据，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        test_size: 测试集比例，默认为 0.2
        random_state: 随机种子，默认为 42
    
    返回:
        包含以下键的字典:
            - model: 训练好的模型
            - r2_score: R² 分数
            - coefficients: 模型系数
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    coefficients = model.coef_
    
    return {
        "model": model,
        "r2_score": r2,
        "coefficients": coefficients
    }
