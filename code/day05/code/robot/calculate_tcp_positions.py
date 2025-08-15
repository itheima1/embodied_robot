#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算机械臂动作序列的末端TCP位置
基于运动学正解算法计算各个关节角度对应的末端姿态坐标
"""

import numpy as np
from inverse_kinematics_solver import InverseKinematicsSolver

def calculate_tcp_positions():
    """计算动作序列中各个位置的TCP坐标"""
    
    # 创建运动学求解器实例
    solver = InverseKinematicsSolver()
    
    # 定义动作序列
    sequences = [
        {
            'name': '位置1',
            'angles_deg': [-2.4, 19.1, 90, 26, -0.5],
            'full_angles': [-2.4, 19.1, 90, 26, -0.5, -49.5]
        },
        {
            'name': '位置2', 
            'angles_deg': [-3.2, 19.1, 90, 25.3, -0.6],
            'full_angles': [-3.2, 19.1, 90, 25.3, -0.6, -73.2]
        },
        {
            'name': '初始位置',
            'angles_deg': [-2.1, -90, 90, 90, -0.8],
            'full_angles': [-2.1, -90, 90, 90, -0.8, -76.4]
        }
    ]
    
    print("关节角度 -> TCP坐标 (x, y, z, r, p, y)")
    print("="*50)
    
    for seq in sequences:
        # 将角度转换为弧度
        angles_rad = np.deg2rad(seq['angles_deg'])
        
        try:
            # 计算正向运动学
            T_matrices, T_final_tcp = solver.forward_kinematics(angles_rad)
            
            # 获取TCP位置
            tcp_position = solver.get_current_tcp_position(angles_rad)
            
            # 提取TCP姿态（旋转矩阵）
            tcp_rotation = T_final_tcp[:3, :3]
            
            # 将旋转矩阵转换为欧拉角（RPY）
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(tcp_rotation)
            tcp_euler = r.as_euler('xyz', degrees=True)  # 转换为度
            
            # 输出结果
            print(f"{seq['name']}:")
            print(f"  关节角度: {seq['full_angles']}")
            print(f"  TCP坐标: x={tcp_position[0]*1000:.3f}, y={tcp_position[1]*1000:.3f}, z={tcp_position[2]*1000:.3f}, r={tcp_euler[0]:.3f}, p={tcp_euler[1]:.3f}, y={tcp_euler[2]:.3f}")
            print()
            
        except Exception as e:
            print(f"{seq['name']}: 计算失败 - {e}")
            print()



if __name__ == "__main__":
    try:
        calculate_tcp_positions()
    except Exception as e:
        print(f"程序执行失败: {e}")