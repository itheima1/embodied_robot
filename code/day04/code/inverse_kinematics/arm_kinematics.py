import numpy as np
import math
from typing import Tuple, List, Optional

class ArmKinematics:
    """
    5自由度机械臂运动学工具类
    基于URDF文件中的关节配置实现正解和逆解
    """
    
    def __init__(self):
        # 根据URDF文件定义的关节参数
        # DH参数表 [a, alpha, d, theta_offset]
        # 注意：这里根据URDF中的origin和axis信息推导DH参数
        
        # 连杆长度 (从URDF的origin xyz值推导)
        self.L1 = 0.0265  # Base到yao的z偏移
        self.L2 = 0.081   # yao到jian1的x偏移
        self.L3 = 0.118   # jian1到jian2的z偏移
        self.L4 = 0.118   # jian2到wan的z偏移
        self.L5 = 0.0635  # wan到wan2的z偏移
        self.L6 = 0.021   # wan2到zhua的z偏移
        
        # 关节限制 (弧度)
        self.joint_limits = [
            (-1.57, 1.57),  # Joint 1: 腰部旋转
            (-1.57, 1.57),  # Joint 2: 肩部
            (-1.57, 1.57),  # Joint 3: 肘部
            (-1.57, 1.57),  # Joint 4: 腕部1
            (-1.57, 1.57),  # Joint 5: 腕部2
        ]
        
    def rotation_matrix_x(self, angle: float) -> np.ndarray:
        """绕X轴旋转矩阵"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def rotation_matrix_y(self, angle: float) -> np.ndarray:
        """绕Y轴旋转矩阵"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def rotation_matrix_z(self, angle: float) -> np.ndarray:
        """绕Z轴旋转矩阵"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def transformation_matrix(self, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """构建4x4变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T
    
    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        正运动学：根据关节角度计算末端执行器位置和姿态
        
        Args:
            joint_angles: 5个关节角度 [theta1, theta2, theta3, theta4, theta5] (弧度)
            
        Returns:
            position: 末端执行器位置 [x, y, z]
            orientation: 末端执行器姿态矩阵 (3x3)
        """
        if len(joint_angles) != 5:
            raise ValueError("需要5个关节角度")
        
        # 检查关节限制
        for i, angle in enumerate(joint_angles):
            if not (self.joint_limits[i][0] <= angle <= self.joint_limits[i][1]):
                print(f"警告: 关节{i+1}角度{angle:.3f}超出限制范围{self.joint_limits[i]}")
        
        theta1, theta2, theta3, theta4, theta5 = joint_angles
        
        # 根据URDF文件构建变换矩阵链
        # Base -> yao (Joint 1)
        T01 = self.transformation_matrix(
            np.array([-0.013, 0, self.L1]),
            self.rotation_matrix_y(-np.pi/2) @ self.rotation_matrix_x(theta1)
        )
        
        # yao -> jian1 (Joint 2)
        T12 = self.transformation_matrix(
            np.array([self.L2, 0, 0]),
            self.rotation_matrix_y(np.pi/2) @ self.rotation_matrix_y(theta2)
        )
        
        # jian1 -> jian2 (Joint 3)
        T23 = self.transformation_matrix(
            np.array([0, 0, self.L3]),
            self.rotation_matrix_y(theta3)
        )
        
        # jian2 -> wan (Joint 4)
        T34 = self.transformation_matrix(
            np.array([0, 0, self.L4]),
            self.rotation_matrix_y(theta4)
        )
        
        # wan -> wan2 (Joint 5)
        T45 = self.transformation_matrix(
            np.array([0, 0, self.L5]),
            self.rotation_matrix_z(theta5)
        )
        
        # wan2 -> zhua (末端执行器)
        T5e = self.transformation_matrix(
            np.array([0, -0.0132, self.L6]),
            np.eye(3)
        )
        
        # 计算总变换矩阵
        T0e = T01 @ T12 @ T23 @ T34 @ T45 @ T5e
        
        position = T0e[:3, 3]
        orientation = T0e[:3, :3]
        
        return position, orientation
    
    def inverse_kinematics(self, target_position: np.ndarray, target_orientation: Optional[np.ndarray] = None, 
                          tolerance: float = 1e-3, max_attempts: int = 10) -> Optional[List[float]]:
        """
        逆运动学：根据目标位置和姿态计算关节角度
        使用改进的多初始值数值迭代方法求解
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标姿态矩阵 (3x3)，可选
            tolerance: 位置误差容忍度 (m)
            max_attempts: 最大尝试次数
            
        Returns:
            joint_angles: 5个关节角度，如果无解返回最佳近似解
        """
        if target_orientation is None:
            target_orientation = np.eye(3)
        
        best_solution = None
        best_error = float('inf')
        
        # 生成多个初始猜测值
        initial_guesses = self._generate_initial_guesses(target_position, max_attempts)
        
        for attempt, initial_angles in enumerate(initial_guesses):
            solution, final_error = self._solve_ik_single_attempt(
                target_position, target_orientation, initial_angles, tolerance
            )
            
            if solution is not None:
                # 找到精确解
                return solution
            
            # 记录最佳近似解
            if final_error < best_error:
                best_error = final_error
                best_solution = initial_angles.copy()
        
        # 如果没有找到精确解，返回最佳近似解
        if best_solution is not None:
            print(f"逆运动学未找到精确解，返回最佳近似解，位置误差: {best_error:.6f} m")
            return best_solution
        
        print("逆运动学求解完全失败")
        return None
    
    def _generate_initial_guesses(self, target_position: np.ndarray, num_guesses: int) -> List[List[float]]:
        """
        生成多个初始猜测值
        """
        guesses = []
        
        # 1. 零位初始猜测
        guesses.append([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 2. 基于目标位置的几何初始猜测
        x, y, z = target_position
        
        # 计算基座旋转角度
        theta1_guess = math.atan2(y, x)
        theta1_guess = np.clip(theta1_guess, self.joint_limits[0][0], self.joint_limits[0][1])
        
        # 简化的几何解
        r = math.sqrt(x**2 + y**2)
        z_adj = z - self.L1
        
        # 尝试不同的肘部配置
        elbow_configs = [1, -1]  # 肘部向上和向下
        
        for elbow_sign in elbow_configs:
            try:
                # 简化的2D逆运动学求解（在垂直平面内）
                L_total = self.L3 + self.L4  # 主要连杆长度
                target_dist = math.sqrt(r**2 + z_adj**2)
                
                if target_dist <= L_total * 1.8:  # 可达性检查
                    # 计算肘部角度
                    cos_elbow = (self.L3**2 + self.L4**2 - target_dist**2) / (2 * self.L3 * self.L4)
                    cos_elbow = np.clip(cos_elbow, -1, 1)
                    theta3_guess = elbow_sign * math.acos(cos_elbow)
                    
                    # 计算肩部角度
                    alpha = math.atan2(z_adj, r)
                    beta = math.atan2(self.L4 * math.sin(theta3_guess), 
                                     self.L3 + self.L4 * math.cos(theta3_guess))
                    theta2_guess = alpha - beta
                    
                    # 腕部角度初始猜测
                    theta4_guess = -(theta2_guess + theta3_guess)  # 保持末端水平
                    theta5_guess = 0.0
                    
                    # 限制在关节范围内
                    guess = [
                        np.clip(theta1_guess, self.joint_limits[0][0], self.joint_limits[0][1]),
                        np.clip(theta2_guess, self.joint_limits[1][0], self.joint_limits[1][1]),
                        np.clip(theta3_guess, self.joint_limits[2][0], self.joint_limits[2][1]),
                        np.clip(theta4_guess, self.joint_limits[3][0], self.joint_limits[3][1]),
                        np.clip(theta5_guess, self.joint_limits[4][0], self.joint_limits[4][1])
                    ]
                    guesses.append(guess)
            except:
                pass
        
        # 3. 随机初始猜测
        np.random.seed(42)  # 固定种子以便重现
        for _ in range(num_guesses - len(guesses)):
            random_guess = []
            for i in range(5):
                low, high = self.joint_limits[i]
                angle = np.random.uniform(low * 0.8, high * 0.8)
                random_guess.append(angle)
            guesses.append(random_guess)
        
        return guesses[:num_guesses]
    
    def _solve_ik_single_attempt(self, target_position: np.ndarray, target_orientation: np.ndarray,
                                initial_angles: List[float], tolerance: float) -> Tuple[Optional[List[float]], float]:
        """
        单次逆运动学求解尝试
        """
        current_angles = initial_angles.copy()
        
        # 迭代参数
        max_iterations = 500
        step_size = 0.1
        min_step_size = 0.001
        step_decay = 0.95
        
        for iteration in range(max_iterations):
            # 计算当前位置和姿态
            current_pos, current_rot = self.forward_kinematics(current_angles)
            
            # 计算位置误差
            pos_error = target_position - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            # 检查收敛（主要关注位置精度）
            if pos_error_norm < tolerance:
                return current_angles, pos_error_norm
            
            # 计算雅可比矩阵（只考虑位置，简化计算）
            jacobian = self._compute_position_jacobian(current_angles)
            
            # 使用阻尼最小二乘法求解
            try:
                damping = 0.01
                JTJ = jacobian.T @ jacobian
                JTJ_damped = JTJ + damping * np.eye(5)
                delta_angles = np.linalg.solve(JTJ_damped, jacobian.T @ pos_error)
                
                # 限制步长
                max_delta = np.max(np.abs(delta_angles))
                if max_delta > 0.5:  # 限制最大角度变化
                    delta_angles *= 0.5 / max_delta
                
                # 更新关节角度
                new_angles = current_angles.copy()
                for i in range(5):
                    new_angles[i] += step_size * delta_angles[i]
                    # 限制在关节范围内
                    new_angles[i] = np.clip(new_angles[i], 
                                           self.joint_limits[i][0], 
                                           self.joint_limits[i][1])
                
                # 检查是否改善
                new_pos, _ = self.forward_kinematics(new_angles)
                new_error = np.linalg.norm(target_position - new_pos)
                
                if new_error < pos_error_norm:
                    current_angles = new_angles
                else:
                    # 如果没有改善，减小步长
                    step_size *= step_decay
                    if step_size < min_step_size:
                        break
                        
            except np.linalg.LinAlgError:
                # 如果矩阵求解失败，尝试随机扰动
                for i in range(5):
                    current_angles[i] += np.random.uniform(-0.05, 0.05)
                    current_angles[i] = np.clip(current_angles[i], 
                                               self.joint_limits[i][0], 
                                               self.joint_limits[i][1])
        
        # 返回最终结果
        final_pos, _ = self.forward_kinematics(current_angles)
        final_error = np.linalg.norm(target_position - final_pos)
        
        return None, final_error
    
    def _compute_position_jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """
        计算位置雅可比矩阵（只考虑位置，不考虑姿态）
        """
        epsilon = 1e-6
        jacobian = np.zeros((3, 5))  # 3DOF位置输出，5个关节
        
        # 当前位置
        current_pos, _ = self.forward_kinematics(joint_angles)
        
        # 对每个关节计算偏导数
        for i in range(5):
            # 正向扰动
            angles_plus = joint_angles.copy()
            angles_plus[i] += epsilon
            pos_plus, _ = self.forward_kinematics(angles_plus)
            
            # 计算偏导数
            jacobian[:, i] = (pos_plus - current_pos) / epsilon
        
        return jacobian
    
    def _compute_jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """
        计算雅可比矩阵（数值微分）
        """
        epsilon = 1e-6
        jacobian = np.zeros((6, 5))  # 6DOF输出，5个关节
        
        # 当前位置和姿态
        current_pos, current_rot = self.forward_kinematics(joint_angles)
        current_state = np.concatenate([current_pos, [current_rot[2, 1] - current_rot[1, 2],
                                                     current_rot[0, 2] - current_rot[2, 0],
                                                     current_rot[1, 0] - current_rot[0, 1]]])
        
        # 对每个关节计算偏导数
        for i in range(5):
            # 正向扰动
            angles_plus = joint_angles.copy()
            angles_plus[i] += epsilon
            pos_plus, rot_plus = self.forward_kinematics(angles_plus)
            state_plus = np.concatenate([pos_plus, [rot_plus[2, 1] - rot_plus[1, 2],
                                                   rot_plus[0, 2] - rot_plus[2, 0],
                                                   rot_plus[1, 0] - rot_plus[0, 1]]])
            
            # 计算偏导数
            jacobian[:, i] = (state_plus - current_state) / epsilon
        
        return jacobian
    
    def check_workspace_limits(self, position: np.ndarray) -> bool:
        """
        检查位置是否在工作空间内
        """
        x, y, z = position
        
        # 改进的工作空间检查
        # 考虑机械臂的实际几何结构
        max_reach = self.L2 + self.L3 + self.L4 + self.L5 + self.L6
        min_reach = max(0, abs(self.L3 + self.L4 + self.L5 + self.L6 - self.L2))
        
        # 计算到基座的距离（考虑基座高度）
        distance = np.sqrt(x**2 + y**2 + (z - self.L1)**2)
        
        # 基本可达性检查
        if not (min_reach <= distance <= max_reach):
            return False
        
        # 高度限制检查
        if z < 0 or z > max_reach + self.L1:
            return False
        
        return True
    
    def get_best_reachable_position(self, target_position: np.ndarray) -> np.ndarray:
        """
        如果目标位置不可达，返回最接近的可达位置
        """
        if self.check_workspace_limits(target_position):
            return target_position
        
        x, y, z = target_position
        
        # 计算到基座的距离
        distance = np.sqrt(x**2 + y**2 + (z - self.L1)**2)
        max_reach = self.L2 + self.L3 + self.L4 + self.L5 + self.L6
        
        if distance > max_reach:
            # 如果超出最大到达距离，缩放到边界
            scale = max_reach / distance
            new_x = x * scale
            new_y = y * scale
            new_z = self.L1 + (z - self.L1) * scale
        else:
            # 如果在最小距离内，推到最小边界
            min_reach = max(0, abs(self.L3 + self.L4 + self.L5 + self.L6 - self.L2))
            if distance < min_reach and distance > 0:
                scale = min_reach / distance
                new_x = x * scale
                new_y = y * scale
                new_z = self.L1 + (z - self.L1) * scale
            else:
                new_x, new_y, new_z = x, y, max(0, z)
        
        return np.array([new_x, new_y, new_z])
    
    def get_joint_limits(self) -> List[Tuple[float, float]]:
        """
        获取关节限制
        """
        return self.joint_limits.copy()
    
    def degrees_to_radians(self, angles_deg: List[float]) -> List[float]:
        """
        角度转弧度
        """
        return [math.radians(angle) for angle in angles_deg]
    
    def radians_to_degrees(self, angles_rad: List[float]) -> List[float]:
        """
        弧度转角度
        """
        return [math.degrees(angle) for angle in angles_rad]