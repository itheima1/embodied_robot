### 实验 1: 线性回归 - 预测数值
# 实验目的: 演示如何使用线性回归模型拟合数据，并进行数值预测。
# 实验输入: 一组模拟的房屋面积和价格数据。
# 实验输出: 数据散点图、拟合的直线、模型的w和b以及对新数据的预测。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 代码开始 ---
# 1. 创建模拟数据 (输入)
# 设置随机种子以保证结果可复现
np.random.seed(0)
# 房屋面积 (特征 X)，单位：平方米
X = np.random.rand(100, 1) * 100 + 50 
# 房屋价格 (标签 y)，单位：万元。y = 5*X + 20 + 噪声
y = 5 * X + 20 + np.random.randn(100, 1) * 50

# 2. 选择并训练模型
model = LinearRegression()
model.fit(X, y)

# 3. 获取模型参数 (输出)
w = model.coef_[0][0]
b = model.intercept_[0]
print(f"模型学习到的权重 (w): {w:.2f}")
print(f"模型学习到的偏置 (b): {b:.2f}")

# 4. 可视化结果 (输出)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='真实数据')
plt.plot(X, model.predict(X), color='red', linewidth=3, label='线性回归拟合线')
plt.title('房屋面积 vs 价格 (线性回归)')
plt.xlabel('面积 (平方米)')
plt.ylabel('价格 (万元)')
plt.legend()
plt.grid(True)
plt.show()

# 5. 用模型进行预测 (输出)
new_area = np.array([[120]]) # 预测一个120平米的房子
predicted_price = model.predict(new_area)
print(f"预测一个面积为 {new_area[0][0]} 平方米的房子，价格大约为: {predicted_price[0][0]:.2f} 万元")
# --- 代码结束 ---