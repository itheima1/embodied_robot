#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂运动学演示程序
简化版本，用于快速测试和演示
"""

import numpy as np
import math
from arm_kinematics import ArmKinematics

def demo_basic_functionality():
    """
    演示基本功能
    """
    print("=" * 60)
    print("机械臂运动学演示程序")
    print("=" * 60)
    
    # 创建机械臂实例
    arm = ArmKinematics()
    
    print("\n1. 机械臂参数信息:")
    print(f"   连杆长度: L1={arm.L1:.4f}m, L2={arm.L2:.4f}m, L3={arm.L3:.4f}m")
    print(f"            L4={arm.L4:.4f}m, L5={arm.L5:.4f}m, L6={arm.L6:.4f}m")
    print(f"   关节限制: {[(f'{math.degrees(low):.0f}°', f'{math.degrees(high):.0f}°') for low, high in arm.joint_limits]}")
    
    # 演示正运动学
    print("\n2. 正运动学演示:")
    test_angles = [30, 45, -30, 20, 60]  # 度
    test_angles_rad = arm.degrees_to_radians(test_angles)
    
    print(f"   输入关节角度: {[f'{a:.0f}°' for a in test_angles]}")
    
    position, orientation = arm.forward_kinematics(test_angles_rad)
    print(f"   末端位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] m")
    print(f"   工作空间检查: {'✓ 在范围内' if arm.check_workspace_limits(position) else '✗ 超出范围'}")
    
    # 演示逆运动学
    print("\n3. 逆运动学演示:")
    target_position = np.array([0.15, 0.1, 0.25])
    print(f"   目标位置: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] m")
    
    if arm.check_workspace_limits(target_position):
        solution = arm.inverse_kinematics(target_position)
        
        if solution is not None:
            solution_deg = arm.radians_to_degrees(solution)
            print(f"   求解成功!")
            print(f"   关节角度: {[f'{a:.1f}°' for a in solution_deg]}")
            
            # 验证解的准确性
            verify_pos, _ = arm.forward_kinematics(solution)
            error = np.linalg.norm(target_position - verify_pos)
            print(f"   验证位置: [{verify_pos[0]:.4f}, {verify_pos[1]:.4f}, {verify_pos[2]:.4f}] m")
            print(f"   位置误差: {error:.6f} m {'✓' if error < 0.001 else '✗'}")
        else:
            print(f"   求解失败!")
    else:
        print(f"   目标位置超出工作空间范围!")
    
    # 演示多个预设位置
    print("\n4. 预设位置测试:")
    preset_positions = [
        ([0, 0, 0, 0, 0], "零位"),
        ([45, 30, -45, 15, 30], "位置A"),
        ([-30, 60, -60, 30, -45], "位置B"),
        ([90, 0, 0, 0, 90], "位置C"),
    ]
    
    for angles_deg, name in preset_positions:
        angles_rad = arm.degrees_to_radians(angles_deg)
        try:
            pos, _ = arm.forward_kinematics(angles_rad)
            print(f"   {name}: 角度{angles_deg} -> 位置[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        except Exception as e:
            print(f"   {name}: 计算失败 - {e}")
    
    print("\n5. 工作空间边界测试:")
    # 测试一些边界位置
    boundary_tests = [
        np.array([0.3, 0, 0.2]),    # 远距离
        np.array([0.05, 0, 0.1]),   # 近距离
        np.array([0, 0.2, 0.3]),    # 侧方
        np.array([0.1, 0.1, 0.4]),  # 高位置
    ]
    
    for i, pos in enumerate(boundary_tests):
        in_workspace = arm.check_workspace_limits(pos)
        status = "✓ 可达" if in_workspace else "✗ 不可达"
        print(f"   测试点{i+1} [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]: {status}")
        
        if in_workspace:
            solution = arm.inverse_kinematics(pos)
            if solution:
                sol_deg = arm.radians_to_degrees(solution)
                print(f"     -> 关节角度: {[f'{a:.0f}°' for a in sol_deg]}")

def demo_trajectory():
    """
    演示轨迹规划
    """
    print("\n" + "=" * 60)
    print("轨迹规划演示")
    print("=" * 60)
    
    arm = ArmKinematics()
    
    # 定义起点和终点
    start_pos = np.array([0.2, 0.0, 0.25])
    end_pos = np.array([0.1, 0.15, 0.3])
    
    print(f"起点: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}] m")
    print(f"终点: [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}] m")
    
    # 检查起点和终点是否可达
    if not (arm.check_workspace_limits(start_pos) and arm.check_workspace_limits(end_pos)):
        print("起点或终点超出工作空间!")
        return
    
    # 生成直线轨迹
    num_points = 10
    trajectory = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        point = start_pos + t * (end_pos - start_pos)
        trajectory.append(point)
    
    print(f"\n生成{num_points}个轨迹点:")
    
    success_count = 0
    joint_trajectory = []
    
    for i, point in enumerate(trajectory):
        solution = arm.inverse_kinematics(point)
        
        if solution is not None:
            joint_trajectory.append(solution)
            solution_deg = arm.radians_to_degrees(solution)
            print(f"点{i+1:2d}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}] -> {[f'{a:5.1f}°' for a in solution_deg]}")
            success_count += 1
        else:
            print(f"点{i+1:2d}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}] -> 求解失败")
    
    print(f"\n轨迹规划结果: {success_count}/{num_points} 点成功求解")
    
    if success_count > 1:
        # 计算关节运动范围
        joint_trajectory = np.array(joint_trajectory)
        print("\n各关节运动范围:")
        for j in range(5):
            joint_angles = joint_trajectory[:, j]
            min_angle = math.degrees(np.min(joint_angles))
            max_angle = math.degrees(np.max(joint_angles))
            range_angle = max_angle - min_angle
            print(f"关节{j+1}: {min_angle:6.1f}° ~ {max_angle:6.1f}° (范围: {range_angle:5.1f}°)")

def main():
    """
    主演示函数
    """
    try:
        demo_basic_functionality()
        demo_trajectory()
        
        print("\n" + "=" * 60)
        print("演示完成!")
        print("\n提示:")
        print("- 运行 test_kinematics.py 进行完整测试")
        print("- 使用 ArmKinematics 类进行自定义开发")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()