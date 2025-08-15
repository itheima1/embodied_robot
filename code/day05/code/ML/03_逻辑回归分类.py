### 实验 3: 逻辑回归 - 进行分类
# 实验目的: 演示如何使用逻辑回归处理二分类问题，并理解其输出是概率。
# 实验输入: 模拟的学生考试时长和是否通过的数据。
# 实验输出: 逻辑回归预测的概率以及对新数据的分类预测。

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 代码开始 ---
# 1. 创建模拟数据 (输入)
# 学习时长 (小时)
X = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5]).reshape(-1, 1)
# 是否通过 (0=未通过, 1=通过)
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# 2. 选择并训练模型
model_logistic = LogisticRegression()
model_logistic.fit(X, y)

# 3. 获取概率输出 (输出)
# 预测每个样本属于类别1(通过)的概率
probabilities = model_logistic.predict_proba(X)[:, 1]

# 4. 可视化 Sigmoid 函数拟合效果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', zorder=20, label='真实数据 (0=未通过, 1=通过)')
X_test_viz = np.linspace(0, 6, 300).reshape(-1, 1)
y_prob_viz = model_logistic.predict_proba(X_test_viz)[:, 1]
plt.plot(X_test_viz, y_prob_viz, color='red', label='逻辑回归拟合的概率曲线 (Sigmoid)')
plt.axhline(y=0.5, color='green', linestyle='--', label='0.5 决策阈值')
plt.title('学习时长 vs 是否通过考试')
plt.xlabel('学习时长 (小时)')
plt.ylabel('通过概率 / 真实标签')
plt.legend()
plt.grid(True)
plt.show()

# 5. 对新数据进行预测 (输出)
study_hours_new = np.array([[2.6]]) # 一个学习了2.6小时的学生
pass_probability = model_logistic.predict_proba(study_hours_new)[:, 1][0]
prediction = model_logistic.predict(study_hours_new)[0]
print(f"一个学习了 {study_hours_new[0][0]} 小时的学生, 通过考试的概率是: {pass_probability:.2f}")
print(f"根据 0.5 的默认阈值，模型预测他: {'通过' if prediction == 1 else '未通过'}")
# --- 代码结束 ---