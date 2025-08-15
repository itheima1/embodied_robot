# demo_numerical.py
# 描述: 专门演示2-DOF平面机械臂的数值法逆运动学。
# 用户可以通过鼠标点击设置目标，程序会从一个固定的初始姿态开始，
# 迭代地逼近目标，并显示整个迭代路径。

import numpy as np
import matplotlib.pyplot as plt

# --- Matplotlib 全局设置，解决中文乱码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 机械臂参数和初始状态 ---
L1 = 1.0
L2 = 1.0
INITIAL_Q = np.array([0.3, 1.2]) # 固定的初始猜测角度

# --- 核心计算函数 ---
def forward_kinematics(q):
    """正向运动学"""
    end_pos = np.array([
        L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]),
        L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
    ])
    elbow_pos = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])
    return end_pos, elbow_pos

def get_jacobian(q):
    """计算雅可比矩阵"""
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-L1*s1 - L2*s12, -L2*s12], [L1*c1 + L2*c12, L2*c12]])

def ik_numerical(target_pos, initial_q, max_iter=100, tol=1e-4):
    """数值法求解IK"""
    q = np.copy(initial_q)
    history = [forward_kinematics(q)[0]]

    for _ in range(max_iter):
        current_pos, _ = forward_kinematics(q)
        if np.linalg.norm(target_pos - current_pos) < tol:
            return q, history
        
        J = get_jacobian(q)
        delta_q = np.linalg.pinv(J) @ (target_pos - current_pos)
        q += 0.5 * delta_q # 步长0.5
        history.append(forward_kinematics(q)[0])
        
    return None, history

# --- 可视化函数 ---
def plot_arm(ax, q, color='blue', linestyle='-', linewidth=4, label=None, alpha=1.0):
    """绘制机械臂"""
    end_pos, elbow_pos = forward_kinematics(q)
    ax.plot([0, elbow_pos[0]], [0, elbow_pos[1]], color=color, lw=linewidth, linestyle=linestyle, alpha=alpha, label=label)
    ax.plot([elbow_pos[0], end_pos[0]], [elbow_pos[1], end_pos[1]], color=color, lw=linewidth, linestyle=linestyle, alpha=alpha)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.plot(elbow_pos[0], elbow_pos[1], 'ko', markersize=10)
    ax.plot(end_pos[0], end_pos[1], 'ro', markersize=10, fillstyle='none', markeredgewidth=2)

def update_plot(ax, target_pos):
    """更新绘图"""
    ax.clear()
    
    # 绘制目标点
    ax.plot(target_pos[0], target_pos[1], 'rX', markersize=15, label="目标点")
    
    # 绘制初始猜测姿态
    plot_arm(ax, INITIAL_Q, color='gray', linestyle=':', linewidth=3, label='初始猜测', alpha=0.8)

    # 计算并绘制解
    final_q, path = ik_numerical(target_pos, INITIAL_Q)
    
    path_np = np.array(path)
    ax.plot(path_np[:, 0], path_np[:, 1], 'm--', label='迭代路径') # 绘制路径
    
    if final_q is not None:
        ax.set_title("数值法演示: 迭代收敛成功！")
        plot_arm(ax, final_q, color='purple', linewidth=5, label='最终解')
    else:
        ax.set_title("数值法演示: 未收敛到目标")
        # 即使未收敛，也画出它最后停在了哪里
        last_q_attempt = np.copy(INITIAL_Q)
        for i in range(len(path) - 1):
             current_pos, _ = forward_kinematics(last_q_attempt)
             J = get_jacobian(last_q_attempt)
             delta_q = np.linalg.pinv(J) @ (path[i+1] - current_pos)
             last_q_attempt += 0.5 * delta_q
        plot_arm(ax, last_q_attempt, color='red', linestyle='-.', linewidth=3, label='停止位置')


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
    target_pos = np.array([event.xdata, event.ydata])
    update_plot(ax, target_pos)

# --- 主程序 ---
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 8))
    target_pos = np.array([0.8, 1.2])
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    update_plot(ax, target_pos)
    
    plt.show()