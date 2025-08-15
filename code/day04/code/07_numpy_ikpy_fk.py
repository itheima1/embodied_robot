import numpy as np

def create_transform_matrix(theta_deg, length):
    """
    根据给定的角度和长度，创建一个2D齐次变换矩阵 (3x3)。
    这个函数是我们理论知识的直接代码实现。
    """
    # 1. 将角度转换为弧度
    theta_rad = np.deg2rad(theta_deg)
    
    # 2. 计算 sin 和 cos 值
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    
    # 3. 构建3x3齐次变换矩阵
    #    这个矩阵代表“先在原点旋转θ，再沿新的X轴平移length”
    #    这正是 T = Trans(L*c, L*s) * Rot(θ) 的结果
    transform_matrix = np.array([
        [c, -s, length * c],
        [s,  c, length * s],
        [0,  0, 1         ]
    ])
    
    return transform_matrix

# ==================== 主程序 ====================

# 1. 定义机器人参数
L1 = 1.0
L2 = 0.8
theta1_deg = 30
theta2_deg = 45

# 2. 为每个连杆创建其相对于【前一个】连杆的变换矩阵
# T₀¹: 描述连杆1相对于基座{0}的变换
T01 = create_transform_matrix(theta1_deg, L1)

# T₁²: 描述连杆2相对于连杆1末端{1}的变换
T12 = create_transform_matrix(theta2_deg, L2)

# 3. 矩阵连乘，得到最终的变换矩阵
# T₀² = T₀¹ × T1²
# 在 numpy 中，'@' 符号就是矩阵乘法
T02 = T01 @ T12

# 4. 从最终矩阵中提取末端位置
# 位置坐标是最终矩阵的最后一列的前两个元素
position = T02[:2, 2]

# 5. 打印结果
print("--- 纯 NumPy 计算结果 ---")
print("T₀¹ (连杆1的变换矩阵):")
print(np.round(T01, 4))
print("\nT₁² (连杆2的变换矩阵):")
print(np.round(T12, 4))
print("\nT₀² (最终的复合变换矩阵):")
print(np.round(T02, 4))

print("\n--- 最终计算结果 ---")
print(f"末端位置 (X, Y): ({position[0]:.4f}, {position[1]:.4f})")

print("\n--- 理论推导结果 ---")
print("末端位置 (X, Y): (1.0733, 1.2821)")