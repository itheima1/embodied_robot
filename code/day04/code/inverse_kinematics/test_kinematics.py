#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂运动学测试程序
测试正运动学和逆运动学功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from arm_kinematics import ArmKinematics

def test_forward_kinematics():
    """
    测试正运动学
    """
    print("=" * 50)
    print("正运动学测试")
    print("=" * 50)
    
    arm = ArmKinematics()
    
    # 测试用例
    test_cases = [
        [0, 0, 0, 0, 0],  # 零位
        [0.5, 0.3, -0.2, 0.1, 0.4],  # 随机位置1
        [1.0, -0.5, 0.8, -0.3, -0.2],  # 随机位置2
        [math.pi/4, math.pi/6, -math.pi/3, math.pi/4, math.pi/2],  # 特殊角度
    ]
    
    for i, angles in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"关节角度 (弧度): {[f'{a:.3f}' for a in angles]}")
        print(f"关节角度 (度): {[f'{math.degrees(a):.1f}°' for a in angles]}")
        
        try:
            position, orientation = arm.forward_kinematics(angles)
            print(f"末端位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            print(f"工作空间检查: {'在范围内' if arm.check_workspace_limits(position) else '超出范围'}")
            
            # 计算末端姿态的欧拉角（简化表示）
            roll = math.atan2(orientation[2, 1], orientation[2, 2])
            pitch = math.atan2(-orientation[2, 0], math.sqrt(orientation[2, 1]**2 + orientation[2, 2]**2))
            yaw = math.atan2(orientation[1, 0], orientation[0, 0])
            print(f"末端姿态 (RPY): [{math.degrees(roll):.1f}°, {math.degrees(pitch):.1f}°, {math.degrees(yaw):.1f}°]")
            
        except Exception as e:
            print(f"错误: {e}")

def test_inverse_kinematics():
    """
    测试逆运动学
    """
    print("\n" + "=" * 50)
    print("逆运动学测试")
    print("=" * 50)
    
    arm = ArmKinematics()
    
    # 测试用例：目标位置
    target_positions = [
        np.array([0.2, 0.0, 0.3]),   # 前方
        np.array([0.0, 0.2, 0.25]),  # 侧方
        np.array([0.15, 0.15, 0.35]), # 斜前方
        np.array([0.1, -0.1, 0.2]),  # 右后方
    ]
    
    for i, target_pos in enumerate(target_positions):
        print(f"\n测试用例 {i+1}:")
        print(f"目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # 检查目标是否在工作空间内
        if not arm.check_workspace_limits(target_pos):
            print("目标位置超出工作空间范围！")
            continue
        
        # 求解逆运动学
        solution = arm.inverse_kinematics(target_pos)
        
        if solution is not None:
            print(f"求解成功！")
            print(f"关节角度 (弧度): {[f'{a:.3f}' for a in solution]}")
            print(f"关节角度 (度): {[f'{math.degrees(a):.1f}°' for a in solution]}")
            
            # 验证解的正确性
            verify_pos, verify_rot = arm.forward_kinematics(solution)
            error = np.linalg.norm(target_pos - verify_pos)
            print(f"验证位置: [{verify_pos[0]:.4f}, {verify_pos[1]:.4f}, {verify_pos[2]:.4f}]")
            print(f"位置误差: {error:.6f} m")
            
            if error < 0.001:
                print("✓ 解验证通过")
            else:
                print("✗ 解验证失败")
        else:
            print("求解失败！")

def test_round_trip():
    """
    测试往返精度：正解->逆解->正解
    """
    print("\n" + "=" * 50)
    print("往返精度测试")
    print("=" * 50)
    
    arm = ArmKinematics()
    
    # 随机生成测试角度
    np.random.seed(42)  # 固定随机种子以便重现
    
    success_count = 0
    total_tests = 10
    
    for i in range(total_tests):
        # 生成随机关节角度（在限制范围内）
        original_angles = []
        for j in range(5):
            low, high = arm.joint_limits[j]
            angle = np.random.uniform(low * 0.8, high * 0.8)  # 留一些余量
            original_angles.append(angle)
        
        print(f"\n测试 {i+1}/{total_tests}:")
        print(f"原始角度: {[f'{math.degrees(a):.1f}°' for a in original_angles]}")
        
        # 正运动学
        target_pos, target_rot = arm.forward_kinematics(original_angles)
        print(f"目标位置: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        
        # 逆运动学
        solved_angles = arm.inverse_kinematics(target_pos, target_rot)
        
        if solved_angles is not None:
            # 再次正运动学验证
            final_pos, final_rot = arm.forward_kinematics(solved_angles)
            
            pos_error = np.linalg.norm(target_pos - final_pos)
            angle_error = np.linalg.norm(np.array(original_angles) - np.array(solved_angles))
            
            print(f"求解角度: {[f'{math.degrees(a):.1f}°' for a in solved_angles]}")
            print(f"位置误差: {pos_error:.6f} m")
            print(f"角度误差: {math.degrees(angle_error):.3f}°")
            
            if pos_error < 0.001:
                print("✓ 测试通过")
                success_count += 1
            else:
                print("✗ 测试失败")
        else:
            print("✗ 逆运动学求解失败")
    
    print(f"\n总体结果: {success_count}/{total_tests} 测试通过 ({100*success_count/total_tests:.1f}%)")

def visualize_workspace():
    """
    可视化机械臂工作空间
    """
    print("\n" + "=" * 50)
    print("工作空间可视化")
    print("=" * 50)
    
    arm = ArmKinematics()
    
    # 生成随机关节角度并计算对应的末端位置
    positions = []
    np.random.seed(42)
    
    for _ in range(1000):
        angles = []
        for j in range(5):
            low, high = arm.joint_limits[j]
            angle = np.random.uniform(low, high)
            angles.append(angle)
        
        try:
            pos, _ = arm.forward_kinematics(angles)
            positions.append(pos)
        except:
            continue
    
    if not positions:
        print("无法生成工作空间点")
        return
    
    positions = np.array(positions)
    
    # 3D散点图
    fig = plt.figure(figsize=(12, 5))
    
    # 3D视图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c=positions[:, 2], cmap='viridis', alpha=0.6, s=1)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('机械臂工作空间 (3D视图)')
    
    # XY平面投影
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(positions[:, 0], positions[:, 1], 
                         c=positions[:, 2], cmap='viridis', alpha=0.6, s=1)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('工作空间 (XY平面投影)')
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.colorbar(scatter, ax=ax2, label='Z (m)')
    plt.tight_layout()
    plt.show()
    
    # 打印工作空间统计信息
    print(f"工作空间统计:")
    print(f"X范围: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
    print(f"Y范围: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
    print(f"Z范围: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")
    print(f"最大到达距离: {np.sqrt((positions**2).sum(axis=1)).max():.3f} m")

def interactive_test():
    """
    交互式测试
    """
    print("\n" + "=" * 50)
    print("交互式测试")
    print("=" * 50)
    
    arm = ArmKinematics()
    
    while True:
        print("\n选择测试模式:")
        print("1. 输入关节角度，计算末端位置 (正运动学)")
        print("2. 输入目标位置，计算关节角度 (逆运动学)")
        print("3. 退出")
        
        try:
            choice = input("请选择 (1-3): ").strip()
            
            if choice == '1':
                print("\n请输入5个关节角度 (度):")
                angles_deg = []
                for i in range(5):
                    while True:
                        try:
                            angle = float(input(f"关节{i+1}角度: "))
                            angles_deg.append(angle)
                            break
                        except ValueError:
                            print("请输入有效数字")
                
                angles_rad = arm.degrees_to_radians(angles_deg)
                position, orientation = arm.forward_kinematics(angles_rad)
                
                print(f"\n结果:")
                print(f"末端位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] m")
                print(f"工作空间检查: {'✓ 在范围内' if arm.check_workspace_limits(position) else '✗ 超出范围'}")
                
            elif choice == '2':
                print("\n请输入目标位置 (m):")
                target_pos = []
                for coord in ['X', 'Y', 'Z']:
                    while True:
                        try:
                            pos = float(input(f"{coord}坐标: "))
                            target_pos.append(pos)
                            break
                        except ValueError:
                            print("请输入有效数字")
                
                target_pos = np.array(target_pos)
                
                if not arm.check_workspace_limits(target_pos):
                    print("警告: 目标位置可能超出工作空间范围")
                
                solution = arm.inverse_kinematics(target_pos)
                
                if solution is not None:
                    print(f"\n求解成功!")
                    solution_deg = arm.radians_to_degrees(solution)
                    print(f"关节角度 (度): {[f'{a:.1f}°' for a in solution_deg]}")
                    
                    # 验证
                    verify_pos, _ = arm.forward_kinematics(solution)
                    error = np.linalg.norm(target_pos - verify_pos)
                    print(f"验证位置: [{verify_pos[0]:.4f}, {verify_pos[1]:.4f}, {verify_pos[2]:.4f}] m")
                    print(f"位置误差: {error:.6f} m")
                else:
                    print("求解失败! 目标位置可能无法到达")
                    
            elif choice == '3':
                print("退出测试程序")
                break
            else:
                print("无效选择，请重试")
                
        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")

def main():
    """
    主测试函数
    """
    print("机械臂运动学测试程序")
    print("基于5自由度机械臂URDF配置")
    
    # 运行所有测试
    test_forward_kinematics()
    test_inverse_kinematics()
    test_round_trip()
    
    # 询问是否进行可视化和交互测试
    try:
        if input("\n是否进行工作空间可视化? (y/n): ").lower().startswith('y'):
            visualize_workspace()
        
        if input("\n是否进行交互式测试? (y/n): ").lower().startswith('y'):
            interactive_test()
    except KeyboardInterrupt:
        print("\n程序结束")

if __name__ == "__main__":
    main()