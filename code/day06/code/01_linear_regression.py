# -*- coding: utf-8 -*-
"""
Scikit-learn 线性回归示例
作者: 学习者
日期: 2024

本示例演示了如何使用scikit-learn进行线性回归分析，包括：
1. 数据生成和准备
2. 模型训练
3. 预测和评估
4. 结果可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def generate_sample_data(n_samples=100, noise=10):
    """
    生成示例数据
    
    参数:
    n_samples: 样本数量
    noise: 噪声水平
    
    返回:
    X: 特征数据
    y: 目标变量
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 生成特征数据 (房屋面积)
    X = np.random.uniform(50, 200, n_samples).reshape(-1, 1)
    
    # 生成目标变量 (房价) - 线性关系加噪声
    # 假设房价 = 面积 * 500 + 10000 + 噪声
    y = X.flatten() * 500 + 10000 + np.random.normal(0, noise * 1000, n_samples)
    
    return X, y

def train_linear_regression(X_train, y_train):
    """
    训练线性回归模型
    
    参数:
    X_train: 训练特征
    y_train: 训练目标
    
    返回:
    model: 训练好的模型
    """
    # 创建线性回归模型
    model = LinearRegression()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    X_test: 测试特征
    y_test: 测试目标
    
    返回:
    predictions: 预测结果
    mse: 均方误差
    r2: R²分数
    """
    # 进行预测
    predictions = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return predictions, mse, r2

def plot_results(X, y, X_test, y_test, predictions, model):
    """
    可视化结果
    
    参数:
    X: 所有特征数据
    y: 所有目标数据
    X_test: 测试特征
    y_test: 测试目标
    predictions: 预测结果
    model: 训练好的模型
    """
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, alpha=0.6, color='blue', label='所有数据点')
    plt.xlabel('房屋面积 (平方米)')
    plt.ylabel('房价 (元)')
    plt.title('原始数据分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(X_test, y_test, alpha=0.6, color='blue', label='实际值')
    plt.scatter(X_test, predictions, alpha=0.6, color='red', label='预测值')
    plt.xlabel('房屋面积 (平方米)')
    plt.ylabel('房价 (元)')
    plt.title('测试集：实际值 vs 预测值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # 绘制回归线
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)
    plt.scatter(X, y, alpha=0.6, color='lightblue', label='训练数据')
    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='回归线')
    plt.xlabel('房屋面积 (平方米)')
    plt.ylabel('房价 (元)')
    plt.title('线性回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # 残差图
    residuals = y_test - predictions
    plt.scatter(predictions, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数
    """
    print("=" * 50)
    print("Scikit-learn 线性回归示例")
    print("=" * 50)
    
    # 1. 生成示例数据
    print("\n1. 生成示例数据...")
    X, y = generate_sample_data(n_samples=100, noise=10)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 2. 划分训练集和测试集
    print("\n2. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 3. 训练模型
    print("\n3. 训练线性回归模型...")
    model = train_linear_regression(X_train, y_train)
    
    # 输出模型参数
    print(f"模型系数 (斜率): {model.coef_[0]:.2f}")
    print(f"模型截距: {model.intercept_:.2f}")
    print(f"回归方程: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
    
    # 4. 模型评估
    print("\n4. 模型评估...")
    predictions, mse, r2 = evaluate_model(model, X_test, y_test)
    
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {np.sqrt(mse):.2f}")
    print(f"R² 分数: {r2:.4f}")
    
    # 5. 结果解释
    print("\n5. 结果解释:")
    print(f"- R² = {r2:.4f} 表示模型能解释 {r2*100:.2f}% 的数据变异")
    if r2 > 0.8:
        print("- 模型拟合效果很好")
    elif r2 > 0.6:
        print("- 模型拟合效果较好")
    else:
        print("- 模型拟合效果一般，可能需要更复杂的模型")
    
    # 6. 进行新预测
    print("\n6. 进行新预测...")
    new_areas = np.array([[80], [120], [150]])
    new_predictions = model.predict(new_areas)
    
    print("新房屋面积预测:")
    for area, price in zip(new_areas.flatten(), new_predictions):
        print(f"  面积 {area} 平方米 -> 预测房价 {price:.0f} 元")
    
    # 7. 可视化结果
    print("\n7. 生成可视化图表...")
    plot_results(X, y, X_test, y_test, predictions, model)
    
    # 8. 保存结果到CSV
    print("\n8. 保存结果...")
    results_df = pd.DataFrame({
        '面积': X_test.flatten(),
        '实际房价': y_test,
        '预测房价': predictions,
        '误差': y_test - predictions
    })
    results_df.to_csv('linear_regression_results.csv', index=False, encoding='utf-8-sig')
    print("结果已保存到 'linear_regression_results.csv'")
    
    print("\n" + "=" * 50)
    print("线性回归示例完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()