import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# =============================================================================
# 1. 机器人模型 (与之前相同)
# =============================================================================
ROBOT_STRUCTURE = [
    {'origin_xyz': np.array([-0.013, 0, 0.0265]), 'origin_rpy': np.array([0, -1.57, 0]), 'axis': np.array([1, 0, 0])},
    {'origin_xyz': np.array([0.081, 0, 0.0]), 'origin_rpy': np.array([0, 1.57, 0]), 'axis': np.array([0, 1, 0])},
    {'origin_xyz': np.array([0, 0, 0.118]), 'origin_rpy': np.array([0, 0, 0]), 'axis': np.array([0, 1, 0])},
    {'origin_xyz': np.array([0, 0, 0.118]), 'origin_rpy': np.array([0, 0, 0]), 'axis': np.array([0, 1, 0])},
    {'origin_xyz': np.array([0, 0, 0.0635]), 'origin_rpy': np.array([0, 0, 0]), 'axis': np.array([0, 0, 1])}
]
TCP_OFFSET_XYZ = np.array([0, 0, 0.05])
TCP_OFFSET_RPY = np.array([0, 0, 0])

# --- 新增：定义关节限位 (根据您的描述) ---
JOINT_LIMIT_MIN = np.deg2rad(np.array([-90.0] * 5))
JOINT_LIMIT_MAX = np.deg2rad(np.array([90.0] * 5))
# 夹爪限位
GRIPPER_LIMIT_MIN = 0.0
GRIPPER_LIMIT_MAX = np.deg2rad(90.0)


def get_flange_to_tcp_transform():
    T_tcp = np.identity(4)
    T_tcp[:3, :3] = R.from_euler('xyz', TCP_OFFSET_RPY).as_matrix()
    T_tcp[:3, 3] = TCP_OFFSET_XYZ
    return T_tcp

T_FLANGE_TO_TCP = get_flange_to_tcp_transform()

# (正向运动学和雅可比矩阵计算函数与上一版相同，这里省略以保持简洁)
def get_transform_matrix_from_urdf(xyz, rpy, axis, angle):
    T_origin_pos = np.identity(4)
    T_origin_pos[:3, 3] = xyz
    T_origin_rot = np.identity(4)
    T_origin_rot[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T_fixed = T_origin_pos @ T_origin_rot
    T_rotation = np.identity(4)
    T_rotation[:3, :3] = R.from_rotvec(angle * axis).as_matrix()
    return T_fixed @ T_rotation

def forward_kinematics(q_5dof):
    T_matrices = [np.identity(4)]
    T_current = np.identity(4)
    for i in range(5):
        joint_info = ROBOT_STRUCTURE[i]
        T_i = get_transform_matrix_from_urdf(joint_info['origin_xyz'], joint_info['origin_rpy'], joint_info['axis'], q_5dof[i])
        T_current = T_current @ T_i
        T_matrices.append(T_current)
    T_final_flange = T_current
    T_final_tcp = T_final_flange @ T_FLANGE_TO_TCP
    return T_matrices, T_final_tcp

def get_position_jacobian(q_5dof):
    T_matrices, T_final_tcp = forward_kinematics(q_5dof)
    p_end = T_final_tcp[:3, 3]
    J_p = np.zeros((3, 5))
    for i in range(5):
        joint_info = ROBOT_STRUCTURE[i]
        T_i = T_matrices[i]
        rotation_axis_base = T_i[:3, :3] @ joint_info['axis']
        p_i = T_i[:3, 3]
        J_p[:, i] = np.cross(rotation_axis_base, p_end - p_i)
    return J_p

# =============================================================================
# 2. 修正后的模拟机器人控制器 (增加了限位检查)
# =============================================================================
class SimulatedRobot:
    def __init__(self, initial_q_all):
        self.current_q = np.array(initial_q_all, dtype=float)
        print("模拟机器人已初始化。")
        self.print_state()

    def get_joint_angles_5dof(self):
        return self.current_q[:5]

    def set_joint_angles_5dof(self, new_q_5dof):
        # --- 新增：安全检查 - 关节限位 ---
        if np.any(new_q_5dof < JOINT_LIMIT_MIN) or np.any(new_q_5dof > JOINT_LIMIT_MAX):
            # print("警告: 计算出的目标角度超出关节限位！移动中止。")
            # print(f"  目标(deg): {np.rad2deg(new_q_5dof)}")
            return False # 表示设置失败
            
        self.current_q[:5] = new_q_5dof
        return True
    
    def set_gripper_angle_deg(self, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        self.current_q[5] = np.clip(angle_rad, GRIPPER_LIMIT_MIN, GRIPPER_LIMIT_MAX)
        print(f"夹爪角度设置为: {np.rad2deg(self.current_q[5]):.1f} 度")

    def print_state(self):
        q_5dof = self.get_joint_angles_5dof()
        _, T_final_tcp = forward_kinematics(q_5dof)
        pos = T_final_tcp[:3, 3]
        angles_deg = np.rad2deg(q_5dof)
        gripper_deg = np.rad2deg(self.current_q[5])
        print(f"  当前关节角度(deg): [{angles_deg[0]:.1f}, {angles_deg[1]:.1f}, {angles_deg[2]:.1f}, {angles_deg[3]:.1f}, {angles_deg[4]:.1f}], 夹爪: {gripper_deg:.1f}")
        print(f"  当前TCP位置(m):  [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# =============================================================================
# 3. 修正后的实时控制逻辑 (使用DLS和速度限制)
# =============================================================================
def move_tcp(robot, direction, step_size_mm=1.0, duration_sec=2.0, damping_factor=0.1, max_joint_speed_rad=0.5):
    direction = np.array(direction) / np.linalg.norm(direction)
    delta_X = direction * (step_size_mm / 1000.0)

    print(f"\n---> 开始移动TCP，方向: {direction} <---")
    
    start_time = time.time()
    steps = 0
    while time.time() - start_time < duration_sec:
        q_5dof = robot.get_joint_angles_5dof()
        Jp = get_position_jacobian(q_5dof)
        
        # --- 核心修正：使用阻尼最小二乘法 (DLS) 代替伪逆 ---
        Jp_T = Jp.T
        lambda_sq = (damping_factor**2) * np.identity(3) # 阻尼项 λ² * I
        try:
            # delta_q = Jp.T * inv(Jp * Jp.T + λ² * I) * delta_X
            inv_term = np.linalg.inv(Jp @ Jp_T + lambda_sq)
            delta_q = Jp_T @ inv_term @ delta_X
        except np.linalg.LinAlgError:
            print("错误: DLS求解失败，矩阵依然奇异。移动中止。")
            break

        # --- 新增：安全检查 - 关节速度限制 ---
        joint_speed_norm = np.linalg.norm(delta_q)
        if joint_speed_norm > max_joint_speed_rad:
            # print(f"警告: 关节速度过高 ({joint_speed_norm:.2f} > {max_joint_speed_rad})，进行缩放。")
            delta_q = delta_q * (max_joint_speed_rad / joint_speed_norm)

        new_q = q_5dof + delta_q
        if not robot.set_joint_angles_5dof(new_q):
            print("警告: 到达关节限位，移动中止。")
            break
        
        steps += 1
        time.sleep(0.02)

    print(f"---> 移动结束 (执行了 {steps} 步) <---\n")
    robot.print_state()

# =============================================================================
# 4. 主程序入口
# =============================================================================
if __name__ == '__main__':
    initial_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    my_robot = SimulatedRobot(initial_angles)
    
    # 移动到一个稍微舒展的初始姿态，以避免从奇异点开始
    print("移动到一个初始工作姿态...")
    my_robot.set_joint_angles_5dof(np.deg2rad([0, 30, -30, -30, 0]))
    my_robot.print_state()
    
    # --- 执行一系列动作 ---
    my_robot.set_gripper_angle_deg(90)
    
    # Damping factor可以调整，值越大越稳定，但末端跟踪精度可能略微下降
    move_tcp(my_robot, direction=[0, 0, 1], damping_factor=0.05)  # 向上
    move_tcp(my_robot, direction=[1, 0, 0], damping_factor=0.05)  # 向前
    
    my_robot.set_gripper_angle_deg(0)
    
    move_tcp(my_robot, direction=[0, 1, 0], damping_factor=0.05)  # 向右
    move_tcp(my_robot, direction=[-1, 0, -1], damping_factor=0.05) # 向左后下方