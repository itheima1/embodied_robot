import numpy as np
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 机械臂反解求解器 - 纯业务逻辑
# =============================================================================

class InverseKinematicsSolver:
    """
    6自由度机械臂反解求解器
    提供末端位置控制的反解计算功能
    """
    
    def __init__(self):
        # 机器人结构参数
        self.ROBOT_STRUCTURE = [
            {'origin_xyz': np.array([-0.013, 0, 0.0265]), 'origin_rpy': np.array([0, -1.57, 0]), 'axis': np.array([1, 0, 0])},
            {'origin_xyz': np.array([0.081, 0, 0.0]), 'origin_rpy': np.array([0, 1.57, 0]), 'axis': np.array([0, 1, 0])},
            {'origin_xyz': np.array([0, 0, 0.118]), 'origin_rpy': np.array([0, 0, 0]), 'axis': np.array([0, 1, 0])},
            {'origin_xyz': np.array([0, 0, 0.118]), 'origin_rpy': np.array([0, 0, 0]), 'axis': np.array([0, 1, 0])},
            {'origin_xyz': np.array([0, 0, 0.0635]), 'origin_rpy': np.array([0, 0, 0]), 'axis': np.array([0, 0, 1])}
        ]
        
        # TCP偏移
        self.TCP_OFFSET_XYZ = np.array([0, 0, 0.107])
        self.TCP_OFFSET_RPY = np.array([0, 0, 0])
        
        # 关节限位 (弧度)
        self.JOINT_LIMIT_MIN = np.deg2rad(np.array([-90.0] * 5))
        self.JOINT_LIMIT_MAX = np.deg2rad(np.array([90.0] * 5))
        
        # 反解参数
        self.step_size_mm = 5.0  # 每步移动5mm
        self.damping_factor = 0.05  # 阻尼因子
        self.max_joint_speed_rad = 0.5  # 最大关节速度
        
        # 计算TCP变换矩阵
        self.T_FLANGE_TO_TCP = self._get_flange_to_tcp_transform()
    
    def _get_flange_to_tcp_transform(self):
        """计算法兰到TCP的变换矩阵"""
        T_tcp = np.identity(4)
        T_tcp[:3, :3] = R.from_euler('xyz', self.TCP_OFFSET_RPY).as_matrix()
        T_tcp[:3, 3] = self.TCP_OFFSET_XYZ
        return T_tcp
    
    def _get_transform_matrix_from_urdf(self, xyz, rpy, axis, angle):
        """根据URDF参数计算变换矩阵"""
        T_origin_pos = np.identity(4)
        T_origin_pos[:3, 3] = xyz
        T_origin_rot = np.identity(4)
        T_origin_rot[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
        T_fixed = T_origin_pos @ T_origin_rot
        T_rotation = np.identity(4)
        T_rotation[:3, :3] = R.from_rotvec(angle * axis).as_matrix()
        return T_fixed @ T_rotation
    
    def forward_kinematics(self, q_5dof):
        """正向运动学计算
        
        Args:
            q_5dof: 5个关节角度 (弧度)
            
        Returns:
            tuple: (变换矩阵列表, 最终TCP变换矩阵)
        """
        T_matrices = [np.identity(4)]
        T_current = np.identity(4)
        
        for i in range(5):
            joint_info = self.ROBOT_STRUCTURE[i]
            T_i = self._get_transform_matrix_from_urdf(
                joint_info['origin_xyz'], 
                joint_info['origin_rpy'], 
                joint_info['axis'], 
                q_5dof[i]
            )
            T_current = T_current @ T_i
            T_matrices.append(T_current)
        
        T_final_flange = T_current
        T_final_tcp = T_final_flange @ self.T_FLANGE_TO_TCP
        return T_matrices, T_final_tcp
    
    def get_position_jacobian(self, q_5dof):
        """计算位置雅可比矩阵
        
        Args:
            q_5dof: 5个关节角度 (弧度)
            
        Returns:
            np.ndarray: 3x5的位置雅可比矩阵
        """
        T_matrices, T_final_tcp = self.forward_kinematics(q_5dof)
        p_end = T_final_tcp[:3, 3]
        J_p = np.zeros((3, 5))
        
        for i in range(5):
            joint_info = self.ROBOT_STRUCTURE[i]
            T_i = T_matrices[i]
            rotation_axis_base = T_i[:3, :3] @ joint_info['axis']
            p_i = T_i[:3, 3]
            J_p[:, i] = np.cross(rotation_axis_base, p_end - p_i)
        
        return J_p
    
    def get_current_tcp_position(self, q_5dof):
        """获取当前TCP位置
        
        Args:
            q_5dof: 5个关节角度 (弧度)
            
        Returns:
            np.ndarray: TCP位置 [x, y, z] (米)
        """
        _, T_final_tcp = self.forward_kinematics(q_5dof)
        return T_final_tcp[:3, 3]
    
    def check_joint_limits(self, q_5dof):
        """检查关节限位
        
        Args:
            q_5dof: 5个关节角度 (弧度)
            
        Returns:
            bool: True表示在限位内，False表示超出限位
        """
        return not (np.any(q_5dof < self.JOINT_LIMIT_MIN) or np.any(q_5dof > self.JOINT_LIMIT_MAX))
    
    def solve_inverse_kinematics_step(self, current_q_5dof, direction):
        """单步反解计算
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            direction: 移动方向 [x, y, z]，会自动归一化
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
                - success (bool): 是否成功
                - new_q_5dof (np.ndarray): 新的关节角度，失败时返回原角度
                - error_msg (str): 错误信息，成功时为空字符串
        """
        try:
            # 归一化方向向量
            direction = np.array(direction, dtype=float)
            if np.linalg.norm(direction) == 0:
                return False, current_q_5dof, "方向向量不能为零向量"
            
            direction = direction / np.linalg.norm(direction)
            
            # 计算位移量 (5mm)
            delta_X = direction * (self.step_size_mm / 1000.0)
            
            # 计算雅可比矩阵
            Jp = self.get_position_jacobian(current_q_5dof)
            
            # 使用阻尼最小二乘法 (DLS) 求解
            Jp_T = Jp.T
            lambda_sq = (self.damping_factor**2) * np.identity(3)
            
            try:
                inv_term = np.linalg.inv(Jp @ Jp_T + lambda_sq)
                delta_q = Jp_T @ inv_term @ delta_X
            except np.linalg.LinAlgError:
                return False, current_q_5dof, "DLS求解失败，矩阵奇异"
            
            # 关节速度限制
            joint_speed_norm = np.linalg.norm(delta_q)
            if joint_speed_norm > self.max_joint_speed_rad:
                delta_q = delta_q * (self.max_joint_speed_rad / joint_speed_norm)
            
            # 计算新的关节角度
            new_q_5dof = current_q_5dof + delta_q
            
            # 检查关节限位
            if not self.check_joint_limits(new_q_5dof):
                return False, current_q_5dof, "超出关节限位"
            
            return True, new_q_5dof, ""
            
        except Exception as e:
            return False, current_q_5dof, f"计算错误: {str(e)}"
    
    def move_forward(self, current_q_5dof):
        """前进 (X+方向)
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
        """
        return self.solve_inverse_kinematics_step(current_q_5dof, [1, 0, 0])
    
    def move_backward(self, current_q_5dof):
        """后退 (X-方向)
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
        """
        return self.solve_inverse_kinematics_step(current_q_5dof, [-1, 0, 0])
    
    def move_left(self, current_q_5dof):
        """左移 (Y+方向)
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
        """
        return self.solve_inverse_kinematics_step(current_q_5dof, [0, 1, 0])
    
    def move_right(self, current_q_5dof):
        """右移 (Y-方向)
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
        """
        return self.solve_inverse_kinematics_step(current_q_5dof, [0, -1, 0])
    
    def move_up(self, current_q_5dof):
        """上升 (Z+方向)
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
        """
        return self.solve_inverse_kinematics_step(current_q_5dof, [0, 0, 1])
    
    def move_down(self, current_q_5dof):
        """下降 (Z-方向)
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
        """
        return self.solve_inverse_kinematics_step(current_q_5dof, [0, 0, -1])
    
    def move_custom_direction(self, current_q_5dof, direction):
        """自定义方向移动
        
        Args:
            current_q_5dof: 当前5个关节角度 (弧度)
            direction: 移动方向 [x, y, z]
            
        Returns:
            tuple: (成功标志, 新的关节角度, 错误信息)
        """
        return self.solve_inverse_kinematics_step(current_q_5dof, direction)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 创建求解器实例
    solver = InverseKinematicsSolver()
    
    # 示例：当前关节角度 (弧度)
    current_angles = np.deg2rad([0, 30, -30, -30, 0])  # 转换为弧度
    
    print("=== 机械臂反解求解器测试 ===")
    print(f"当前关节角度(度): {np.rad2deg(current_angles)}")
    
    # 获取当前TCP位置
    current_pos = solver.get_current_tcp_position(current_angles)
    print(f"当前TCP位置(m): [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
    
    print("\n=== 测试各方向移动 ===")
    
    # 测试前进
    success, new_angles, error = solver.move_forward(current_angles)
    if success:
        new_pos = solver.get_current_tcp_position(new_angles)
        print(f"前进后关节角度(度): {np.rad2deg(new_angles)}")
        print(f"前进后TCP位置(m): [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]")
        print(f"位移量(mm): [{(new_pos[0]-current_pos[0])*1000:.1f}, {(new_pos[1]-current_pos[1])*1000:.1f}, {(new_pos[2]-current_pos[2])*1000:.1f}]")
    else:
        print(f"前进失败: {error}")
    
    # 测试上升
    success, new_angles, error = solver.move_up(current_angles)
    if success:
        new_pos = solver.get_current_tcp_position(new_angles)
        print(f"\n上升后关节角度(度): {np.rad2deg(new_angles)}")
        print(f"上升后TCP位置(m): [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]")
        print(f"位移量(mm): [{(new_pos[0]-current_pos[0])*1000:.1f}, {(new_pos[1]-current_pos[1])*1000:.1f}, {(new_pos[2]-current_pos[2])*1000:.1f}]")
    else:
        print(f"上升失败: {error}")
    
    # 测试自定义方向 (斜向移动)
    success, new_angles, error = solver.move_custom_direction(current_angles, [1, 1, 1])
    if success:
        new_pos = solver.get_current_tcp_position(new_angles)
        print(f"\n斜向移动后关节角度(度): {np.rad2deg(new_angles)}")
        print(f"斜向移动后TCP位置(m): [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]")
        print(f"位移量(mm): [{(new_pos[0]-current_pos[0])*1000:.1f}, {(new_pos[1]-current_pos[1])*1000:.1f}, {(new_pos[2]-current_pos[2])*1000:.1f}]")
    else:
        print(f"斜向移动失败: {error}")
    
    print("\n=== 测试完成 ===")