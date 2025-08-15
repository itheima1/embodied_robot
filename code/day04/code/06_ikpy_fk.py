import ikpy.chain
import ikpy.link
import numpy as np
#  注意ikpy的不同版本api差异很大，我们用的版本是3.4.2  pip install ikpy==3.4.2
# 1. 定义机器人 - 使用 DHLink
my_arm = ikpy.chain.Chain(name='my_2d_arm', links=[
    ikpy.link.OriginLink(),  # 基座链接
    ikpy.link.DHLink(
        name="joint1",
        d=0,        # 连杆偏移
        a=1.0,      # 连杆长度
        alpha=0,    # 连杆扭转角
        theta=0     # 关节角（会被覆盖）
    ),
    ikpy.link.DHLink(
        name="joint2",
        d=0,
        a=0.8,
        alpha=0,
        theta=0
    )
])

# 2. 设置关节角
joint_angles = [0, np.deg2rad(30), np.deg2rad(45)]

# 3. 计算正解
T = my_arm.forward_kinematics(joint_angles)
position = T[:3, 3]

# 4. 打印结果
print("--- IKPY 计算结果 ---")
print(f"末端位置 (X, Y): ({position[0]:.4f}, {position[1]:.4f})")