### 实验 2: 损失函数 - 均方误差(MSE)
# 实验目的: 计算线性回归模型的均方误差(MSE)，量化模型的“好坏”。
# 实验输入: 实验1的数据(X, y)和训练好的模型。
# 实验输出: 打印出的MSE值。

from sklearn.metrics import mean_squared_error

# --- 代码开始 ---
# (请先运行实验1的代码，以获得数据X, y和训练好的model)

# 1. 使用模型进行预测
y_pred = model.predict(X)

# 2. 计算均方误差 (MSE)
mse = mean_squared_error(y, y_pred)

# 3. 输出结果
print(f"该线性回归模型在训练数据上的均方误差 (MSE) 为: {mse:.2f}")
print("这个值代表了模型所有预测值与真实值差距的平方的平均值。我们的目标(通过梯度下降)就是让这个值变得尽可能小。")
# --- 代码结束 ---