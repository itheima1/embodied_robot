# demo_analytical.py
# 描述: 专门演示2-DOF平面机械臂的解析法(几何法)逆运动学。
# 用户可以通过鼠标点击设置目标，程序会立即计算并显示所有可能的精确解。

import numpy as np
import matplotlib.pyplot as plt

# --- Matplotlib 全局设置，解决中文乱码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 机械臂参数 ---
L1 = 1.0  # 连杆1长度
L2 = 1.0  # 连杆2长度

# --- 核心计算函数 ---
def forward_kinematics(q):
    """正向运动学，根据关节角计算末端位置"""
    end_pos = np.array([
        L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]),
        L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
    ])
    elbow_pos = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])
    return end_pos, elbow_pos

def ik_analytical(target_pos):
    """解析法求解IK"""
    x, y = target_pos
    d_squared = x**2 + y**2
    d = np.sqrt(d_squared)

    # 检查可达性 (无解情况)
    if d > L1 + L2 or d < abs(L1 - L2):
        return []

    cos_q2 = np.clip((d_squared - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    
    # 两个可能的q2解
    q2_sol1 = np.arccos(cos_q2)   # "Elbow down"
    q2_sol2 = -np.arccos(cos_q2)  # "Elbow up"
    
    solutions = []
    for q2 in [q2_sol1, q2_sol2]:
        alpha = np.arctan2(y, x)
        beta = np.arccos(np.clip((d_squared + L1**2 - L2**2) / (2 * L1 * d), -1.0, 1.0))
        q1 = alpha - beta if q2 >= 0 else alpha + beta
        solutions.append(np.array([q1, q2]))
        
    return solutions

# --- 可视化函数 ---
def plot_arm(ax, q, color='blue', linestyle='-', label=None):
    """绘制机械臂"""
    _, elbow_pos = forward_kinematics(q)
    ax.plot([0, elbow_pos[0]], [0, elbow_pos[1]], color=color, lw=4, linestyle=linestyle, label=label)
    ax.plot([elbow_pos[0], target_pos[0]], [elbow_pos[1], target_pos[1]], color=color, lw=4, linestyle=linestyle)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.plot(elbow_pos[0], elbow_pos[1], 'ko', markersize=10)

def update_plot(ax, target_pos):
    """更新绘图"""
    ax.clear()
    
    # 绘制工作空间
    workspace_outer = plt.Circle((0, 0), L1 + L2, color='gray', fill=False, linestyle='--', label='最大工作范围')
    ax.add_artist(workspace_outer)

    # 绘制目标点
    ax.plot(target_pos[0], target_pos[1], 'rX', markersize=15, label="目标点")

    # 计算并绘制解
    solutions = ik_analytical(target_pos)
    if not solutions:
        ax.set_title("解析法演示: 目标不可达！")
    else:
        ax.set_title("解析法演示: 已找到所有精确解")
        plot_arm(ax, solutions[0], 'blue', '-', '解 1 (高手肘)')
        # 仅当两解不重合时绘制第二个
        if len(solutions) > 1 and np.linalg.norm(solutions[0] - solutions[1]) > 1e-3:
            plot_arm(ax, solutions[1], 'green', '--', '解 2 (低手肘)')
    
    ax.plot(target_pos[0], target_pos[1], 'ro', markersize=12, fillstyle='none', markeredgewidth=2)
    ax.grid(True)
    ax.set_xlim(-(L1 + L2) * 1.1, (L1 + L2) * 1.1)
    ax.set_ylim(-(L1 + L2) * 1.1, (L1 + L2) * 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')
    ax.text(0.02, 0.02, "请点击以设置新目标", transform=ax.transAxes, fontsize=10, color='gray')
    fig.canvas.draw_idle()

def on_click(event):
    """鼠标点击事件"""
    if event.inaxes != ax: return
    global target_pos
    target_pos = np.array([event.xdata, event.ydata])
    update_plot(ax, target_pos)

# --- 主程序 ---
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 8))
    target_pos = np.array([0.8, 1.2]) # 初始目标
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    update_plot(ax, target_pos)
    
    plt.show()