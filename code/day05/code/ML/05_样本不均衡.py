### 实验 6: 准确率的陷阱 - 样本不均衡
# 实验目的: 演示在样本不均衡时，高准确率的欺骗性。
# 实验输入: 一个模拟的、高度不均衡的邮件数据集。
# 实验输出: 一个“傻瓜”模型和一个真实模型的准确率对比。

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# --- 代码开始 ---
# 1. 创建一个高度不均衡的数据集 (输入)
# 1000个样本, 只有一个特征, 99%是类别0(正常邮件), 1%是类别1(垃圾邮件)
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, n_clusters_per_class=1,
                           weights=[0.99, 0.01], flip_y=0, random_state=1)

# 2. 傻瓜模型：永远预测多数类（正常邮件）
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X, y)
y_pred_dummy = dummy_model.predict(X)
acc_dummy = accuracy_score(y, y_pred_dummy)
print(f"“傻瓜模型”(永远预测正常邮件) 的准确率是: {acc_dummy:.4f}")

# 3. 真实模型：逻辑回归
real_model = LogisticRegression()
real_model.fit(X, y)
y_pred_real = real_model.predict(X)
acc_real = accuracy_score(y, y_pred_real)
print(f"逻辑回归模型的准确率是: {acc_real:.4f}")
print("\n结论: '傻瓜模型'虽然准确率高达99%，但它一个垃圾邮件也找不到，毫无价值。这证明了在样本不均衡时，准确率是不可靠的。")
# --- 代码结束 ---