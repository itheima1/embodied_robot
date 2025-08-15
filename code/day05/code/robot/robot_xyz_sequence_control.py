#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机械臂TCP坐标序列控制程序
执行指定的TCP坐标序列，包括夹爪控制

动作序列：
1. 移动到位置1: TCP坐标 x=257.294, y=-11.328, z=59.620, r=-179.499, p=44.898, y=178.308
2. 执行夹爪闭合
3. 移动到位置2: TCP坐标 x=258.574, y=-15.182, z=61.100, r=-179.384, p=45.597, y=177.660
4. 移动到初始位置: TCP坐标 x=39.529, y=-1.920, z=225.500, r=-89.998, p=89.198, y=-92.098
"""

import serial
import time
import sys
import numpy as np
from inverse_kinematics_solver import InverseKinematicsSolver

# Windows系统UTF-8编码设置
import os
os.system('chcp 65001 > nul')

class ServoController:
    """舵机控制器类"""
    
    def __init__(self, port="COM6", baudrate=1000000, timeout=0.1):
        self.port_name = port
        self.serial_port = None
        # 常量定义
        self.ADDR_GOAL_POSITION = 42
        self.ADDR_PRESENT_POSITION = 56
        self.ADDR_TORQUE_ENABLE = 40
        self.INST_WRITE = 3
        self.INST_READ = 2
        self.COMM_SUCCESS = 0
        self.COMM_RX_TIMEOUT = -6
        self.COMM_RX_CORRUPT = -7
        
        try:
            self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            time.sleep(0.1)
            self.serial_port.reset_input_buffer()
            print(f"成功打开串口 {port}")
        except serial.SerialException as e:
            print(f"错误：无法打开串口 {port}: {e}")
            sys.exit(1)
    
    def close(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"串口 {self.port_name} 已关闭")
    
    def _calculate_checksum(self, data):
        """计算校验和"""
        return (~sum(data)) & 0xFF
    
    def _send_packet(self, servo_id, instruction, parameters=None):
        """发送数据包"""
        if not self.serial_port or not self.serial_port.is_open:
            return False
        if parameters is None:
            parameters = []
        length = len(parameters) + 2
        packet_core = [servo_id, length, instruction] + parameters
        checksum = self._calculate_checksum(packet_core)
        packet = bytes([0xFF, 0xFF] + packet_core + [checksum])
        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.write(packet)
            self.serial_port.flush()
            return True
        except Exception:
            return False
    
    def _read_packet(self):
        """读取数据包"""
        start_time = time.time()
        packet = []
        while (time.time() - start_time) < self.serial_port.timeout:
            if self.serial_port.in_waiting > 0:
                byte = self.serial_port.read(1)
                if not byte: continue
                byte = byte[0]

                if not packet and byte != 0xFF:
                    continue
                
                packet.append(byte)

                if len(packet) >= 2 and packet[-2:] == [0xFF, 0xFF]:
                    if len(packet) > 2:
                        packet = [0xFF, 0xFF]
                    continue

                if len(packet) > 4:
                    pkt_len = packet[3]
                    if len(packet) == pkt_len + 4:
                        core_data = packet[2:-1]
                        calculated_checksum = self._calculate_checksum(core_data)
                        if calculated_checksum == packet[-1]:
                            return self.COMM_SUCCESS, packet[4], packet[5:-1]
                        else:
                            return self.COMM_RX_CORRUPT, 0, []
        return self.COMM_RX_TIMEOUT, 0, []
    
    def _write_register(self, servo_id, address, value, size=2):
        """写寄存器"""
        params = [address]
        if size == 1:
            params.append(value & 0xFF)
        elif size == 2:
            params.extend([value & 0xFF, (value >> 8) & 0xFF])
        else:
            return False
        
        return self._send_packet(servo_id, self.INST_WRITE, params)
    
    def enable_torque(self, servo_id):
        """启用舵机扭矩"""
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, 1, size=1)
    
    def set_servo_angle(self, servo_id, angle):
        """设置舵机角度 (-90到90度)"""
        # 将角度(-90到90)映射到位置(1024到3072)
        position = int(((angle + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        # 限制范围
        position = max(1024, min(3072, position))
        
        return self._write_register(servo_id, self.ADDR_GOAL_POSITION, position, size=2)
    
    def set_all_servo_angles(self, angles):
        """设置所有舵机角度"""
        success = True
        # 根据角度数组长度决定控制的舵机数量
        num_servos = min(len(angles), 6)  # 最多控制6个舵机
        
        for i, angle in enumerate(angles[:num_servos]):
            if not self.set_servo_angle(i + 1, angle):
                success = False
                print(f"设置舵机 {i+1} 角度失败")
            time.sleep(0.02)  # 增加延时避免通信冲突
        return success
    
    def control_gripper(self, close=True):
        """控制夹爪
        
        Args:
            close (bool): True为闭合，False为张开
        """
        # 假设夹爪是第6个舵机
        gripper_id = 6
        if close:
            # 夹爪闭合角度
            angle = -45  # 可根据实际情况调整
            print("执行夹爪闭合...")
        else:
            # 夹爪张开角度
            angle = 45   # 可根据实际情况调整
            print("执行夹爪张开...")
        
        return self.set_servo_angle(gripper_id, angle)

class RobotXYZSequenceController:
    """机械臂TCP坐标序列控制器"""
    
    def __init__(self, port="COM6"):
        # 初始化舵机控制器
        self.servo_controller = ServoController(port=port)
        
        # 初始化逆运动学求解器
        self.ik_solver = InverseKinematicsSolver()
        
        # 启用所有舵机扭矩（包括夹爪）
        print("正在启用舵机扭矩...")
        for servo_id in range(1, 7):  # 1-6号舵机
            self.servo_controller.enable_torque(servo_id)
            time.sleep(0.05)
        print("舵机扭矩已启用")
        
        # 定义TCP坐标动作序列
        self.sequence = [
            {
                'name': '移动到位置1',
                'tcp_coords': [257.294, -11.328, 59.620, -179.499, 44.898, 178.308],
                'wait_time': 3.0
            },
            {
                'name': '夹爪闭合',
                'action': 'gripper_close',
                'wait_time': 2.0
            },
            {
                'name': '移动到位置2',
                'tcp_coords': [258.574, -15.182, 61.100, -179.384, 45.597, 177.660],
                'wait_time': 3.0
            },
            {
                'name': '移动到初始位置',
                'tcp_coords': [39.529, -1.920, 225.500, -89.998, 89.198, -92.098],
                'wait_time': 3.0
            }
        ]
    
    def solve_ik_to_position(self, target_pos, max_iterations=1000, tolerance=0.001):
        """使用迭代方法求解逆运动学到达目标位置
        
        Args:
            target_pos (list): 目标位置 [x, y, z] (米)
            max_iterations (int): 最大迭代次数
            tolerance (float): 位置容差 (米)
            
        Returns:
            tuple: (成功标志, 关节角度列表)
        """
        # 初始关节角度 (弧度)
        current_q = np.deg2rad([0, 0, 0, 0, 0])  # 5个关节
        target_pos = np.array(target_pos) / 1000.0  # 转换为米
        
        for iteration in range(max_iterations):
            # 获取当前TCP位置
            current_pos = self.ik_solver.get_current_tcp_position(current_q)
            
            # 计算位置误差
            position_error = target_pos - current_pos
            error_norm = np.linalg.norm(position_error)
            
            # 检查是否达到容差
            if error_norm < tolerance:
                # 转换为度并返回6个关节角度（第6个关节设为0）
                joint_angles_deg = np.rad2deg(current_q).tolist() + [0]  # 添加第6个关节
                return True, joint_angles_deg
            
            # 使用自定义方向移动
            success, new_q, error_msg = self.ik_solver.move_custom_direction(current_q, position_error)
            
            if not success:
                print(f"迭代 {iteration}: {error_msg}")
                break
            
            current_q = new_q
        
        return False, None
    
    def move_to_tcp_coords(self, tcp_coords, description=""):
        """移动到指定TCP坐标
        
        Args:
            tcp_coords (list): TCP坐标 [x, y, z, r, p, y]
            description (str): 动作描述
        """
        print(f"\n{description}")
        print(f"目标TCP坐标: x={tcp_coords[0]:.3f}, y={tcp_coords[1]:.3f}, z={tcp_coords[2]:.3f}, r={tcp_coords[3]:.3f}, p={tcp_coords[4]:.3f}, y={tcp_coords[5]:.3f}")
        
        # 使用逆运动学求解器计算关节角度（只考虑位置，暂时忽略姿态）
        try:
            target_position = [tcp_coords[0], tcp_coords[1], tcp_coords[2]]  # x, y, z
            success, joint_angles = self.solve_ik_to_position(target_position)
            
            if not success:
                print(f"✗ {description} - 逆运动学求解失败")
                return False
            
            print(f"计算得到的关节角度: {[round(angle, 1) for angle in joint_angles]}")
            
            # 检查角度范围
            for i, angle in enumerate(joint_angles):
                if angle < -90 or angle > 90:
                    print(f"警告：关节{i+1}角度 {angle:.1f}° 超出范围 [-90°, 90°]")
            
            # 发送角度到舵机
            if self.servo_controller.set_all_servo_angles(joint_angles):
                print(f"✓ {description} - TCP坐标设置成功")
                return True
            else:
                print(f"✗ {description} - 舵机角度设置失败")
                return False
                
        except Exception as e:
            print(f"✗ {description} - 运动学计算错误: {e}")
            return False
    
    def execute_sequence(self):
        """执行完整的动作序列"""
        print("\n=== 开始执行机械臂TCP坐标动作序列 ===")
        
        for i, step in enumerate(self.sequence, 1):
            print(f"\n--- 步骤 {i}: {step['name']} ---")
            
            success = False
            
            if 'tcp_coords' in step:
                # TCP坐标移动
                success = self.move_to_tcp_coords(step['tcp_coords'], step['name'])
            elif 'action' in step:
                # 特殊动作（如夹爪控制）
                if step['action'] == 'gripper_close':
                    success = self.servo_controller.control_gripper(close=True)
                    if success:
                        print("✓ 夹爪闭合成功")
                    else:
                        print("✗ 夹爪闭合失败")
                elif step['action'] == 'gripper_open':
                    success = self.servo_controller.control_gripper(close=False)
                    if success:
                        print("✓ 夹爪张开成功")
                    else:
                        print("✗ 夹爪张开失败")
            
            if not success:
                print(f"步骤 {i} 执行失败，继续执行下一步...")
            
            # 等待指定时间
            wait_time = step.get('wait_time', 2.0)
            print(f"等待 {wait_time} 秒...")
            time.sleep(wait_time)
        
        print("\n=== TCP坐标动作序列执行完成 ===")
    
    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        self.servo_controller.close()
        print("程序结束")

def main():
    """主函数"""
    # 配置参数
    STUDENT_PORT = "COM6"  # 根据实际情况调整串口
    
    try:
        # 创建TCP坐标序列控制器
        controller = RobotXYZSequenceController(port=STUDENT_PORT)
        
        # 执行动作序列
        controller.execute_sequence()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行失败: {e}")
        print("请检查：")
        print("1. 串口连接是否正确")
        print("2. 舵机是否正常供电")
        print("3. TCP坐标是否在工作空间内")
        print("4. 逆运动学求解器是否正常工作")
    finally:
        try:
            controller.cleanup()
        except:
            pass

if __name__ == '__main__':
    main()