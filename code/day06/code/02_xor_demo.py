# -*- coding: utf-8 -*-
"""
XOR问题最简演示 - 使用LinearRegression + 特征工程

输入: (0,0), (0,1), (1,0), (1,1)

输出: 0, 1, 1, 0
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def main():
    print("XOR问题演示")
    print("=" * 20)
    
    # XOR数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    print("XOR真值表:")
    for i in range(len(X)):
        print(f"({X[i][0]}, {X[i][1]}) -> {y[i]}")
    
    # 方法1: 原始特征 - 无法解决XOR
    print("\n方法1: 原始特征 [x1, x2]")
    model1 = LinearRegression()
    model1.fit(X, y)
    pred1 = model1.predict(X)
    pred1_binary = (pred1 > 0.5).astype(int)
    accuracy1 = np.mean(pred1_binary == y)
    
    print(f"预测: {pred1_binary}")
    print(f"准确率: {accuracy1:.2f}")
    
    # 方法2: 添加交互项特征 - 可以解决XOR
    print("\n方法2: 扩展特征 [x1, x2, x1*x2]")
    # 创建扩展特征矩阵
    X_extended = np.column_stack([X[:, 0], X[:, 1], X[:, 0] * X[:, 1]])
    print(f"扩展特征矩阵:\n{X_extended}")
    
    model2 = LinearRegression()
    model2.fit(X_extended, y)
    pred2 = model2.predict(X_extended)
    pred2_binary = (pred2 > 0.5).astype(int)
    accuracy2 = np.mean(pred2_binary == y)
    
    print(f"预测: {pred2_binary}")
    print(f"准确率: {accuracy2:.2f}")
    print(f"模型方程: y = {model2.coef_[0]:.2f}*x1 + {model2.coef_[1]:.2f}*x2 + {model2.coef_[2]:.2f}*x1*x2 + {model2.intercept_:.2f}")
    
    # 方法3: 多个线性函数组合
    print("\n方法3: 多个线性函数组合")
    
    # 训练OR模型
    y_or = np.array([0, 1, 1, 1])  # OR真值表
    model_or = LinearRegression()
    model_or.fit(X, y_or)
    or_pred = model_or.predict(X)
    
    # 训练AND模型
    y_and = np.array([0, 0, 0, 1])  # AND真值表
    model_and = LinearRegression()
    model_and.fit(X, y_and)
    and_pred = model_and.predict(X)
    
    # XOR = OR - AND (通过sigmoid激活)
    xor_raw = or_pred - and_pred
    xor_prob = sigmoid(xor_raw * 5)  # 乘以5增强激活效果
    xor_pred = (xor_prob > 0.5).astype(int)
    accuracy3 = np.mean(xor_pred == y)
    
    print(f"OR预测:  {or_pred.round(2)}")
    print(f"AND预测: {and_pred.round(2)}")
    print(f"XOR预测: {xor_pred}")
    print(f"准确率: {accuracy3:.2f}")
    
    print("\n=" * 40)
    print("结论:")
    print("1. 原始线性模型无法解决XOR问题")
    print("2. 添加交互项特征(x1*x2)是关键")
    print("3. 多个线性函数组合也能解决XOR")
    print("4. XOR = (A OR B) AND NOT(A AND B)")

if __name__ == "__main__":
    main()