### 实验 5: 分类模型的损失函数 - 交叉熵
# 实验目的: 直观感受交叉熵损失函数对错误预测的“惩罚”机制。
# 实验输入: 真实标签和两个不同置信度的预测概率。
# 实验输出: 两种情况下计算出的交叉熵损失值。

from sklearn.metrics import log_loss
import numpy as np

# --- 代码开始 ---
# 真实标签: 假设真实类别是“猫”，标签为1
y_true = np.array([1]) # 1代表是猫

# 所有可能的类别标签
all_classes = [0, 1] # 0代表“不是猫”，1代表“是猫”

# 场景1: 模型自信且正确 (输入)
# 模型预测是“猫”的概率为0.99
y_pred_good = np.array([[0.01, 0.99]]) # [非猫概率, 是猫概率]
# 通过 labels 参数告诉函数，这是一个关于类别 0 和 1 的问题
loss_good = log_loss(y_true, y_pred_good, labels=all_classes) # <--- 修改点
print(f"场景1: 真实为'猫', 模型预测'是猫'的概率为0.99")
print(f"交叉熵损失: {loss_good:.4f} (损失非常小)")
print("-" * 30)

# 场景2: 模型自信但完全错误 (输入)
# 模型预测是“猫”的概率仅为0.01 (即认为99%不是猫)
y_pred_bad = np.array([[0.99, 0.01]]) # [非猫概率, 是猫概率]
# 同样需要提供 labels 参数
loss_bad = log_loss(y_true, y_pred_bad, labels=all_classes) # <--- 修改点
print(f"场景2: 真实为'猫', 模型预测'是猫'的概率仅为0.01")
print(f"交叉熵损失: {loss_bad:.4f} (损失非常巨大!)")
print("\n结论: 交叉熵对'猜错'且'非常自信地猜错'的行为给予了巨大的惩罚, 这会激励模型不仅要猜对，还要自信地猜对。")
# --- 代码结束 ---