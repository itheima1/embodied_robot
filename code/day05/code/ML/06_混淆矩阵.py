### 实验 7: 混淆矩阵 - 更精细的评估
# 实验目的: 使用混淆矩阵来详细分析分类模型的性能。
# 实验输入: 实验6中的不均衡数据(X, y)和训练好的逻辑回归模型(real_model)。
# 实验输出: 一个可视化的混淆矩阵。

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# --- 代码开始 ---
# 1. 创建一个高度不均衡的数据集 (输入)
# 1000个样本, 只有一个特征, 99%是类别0(正常邮件), 1%是类别1(垃圾邮件)
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, n_clusters_per_class=1,
                           weights=[0.99, 0.01], flip_y=0, random_state=1)


# 3. 真实模型：逻辑回归
real_model = LogisticRegression()
real_model.fit(X, y)

# 1. 获取模型预测结果
y_pred = real_model.predict(X)

# 2. 计算混淆矩阵
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

# 3. 可视化混淆矩阵 (输出)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测: 正常', '预测: 垃圾'], 
            yticklabels=['真实: 正常', '真实: 垃圾'])
plt.title('混淆矩阵')
plt.ylabel('真实情况')
plt.xlabel('模型预测')
plt.show()

print("混淆矩阵解读 (以垃圾邮件为正例):")
print(f" - TP (True Positive)  真阳性: {tp}  (真实是垃圾邮件, 模型也预测是垃圾邮件 '找对了')")
print(f" - FN (False Negative) 假阴性: {fn}  (真实是垃圾邮件, 模型却预测是正常邮件 '漏报了!')")
print(f" - FP (False Positive) 假阳性: {fp}  (真实是正常邮件, 模型却预测是垃圾邮件 '误报了!')")
print(f" - TN (True Negative)  真阴性: {tn}  (真实是正常邮件, 模型也预测是正常邮件 '找对了')")
# --- 代码结束 ---