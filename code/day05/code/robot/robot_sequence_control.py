#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机械臂连续动作序列控制程序
执行指定的关节角度序列，包括夹爪控制

动作序列：
1. 移动到位置1: [-2.4, 19.1, 90, 26, -0.5, -49.5]
2. 执行夹爪闭合
3. 移动到位置2: [-3.2, 19.1, 90, 25.3, -0.6, -73.2]
4. 移动到初始位置: [-2.1, -90, 90, 90, -0.8, -76.4]
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

class RobotSequenceController:
    """机械臂序列控制器"""
    
    def __init__(self, port="COM6"):
        # 初始化舵机控制器
        self.servo_controller = ServoController(port=port)
        
        # 启用所有舵机扭矩（包括夹爪）
        print("正在启用舵机扭矩...")
        for servo_id in range(1, 7):  # 1-6号舵机
            self.servo_controller.enable_torque(servo_id)
            time.sleep(0.05)
        print("舵机扭矩已启用")
        
        # 定义动作序列
        self.sequence = [
            {
                'name': '移动到位置1',
                'angles': [-2.4, 19.1, 90, 26, -0.5, -49.5],
                'wait_time': 1.0  # 等待3秒
            },
            {
                'name': '夹爪闭合',
                'action': 'gripper_close',
                'wait_time': 1.0  # 等待2秒
            },
            {
                'name': '移动到位置2',
                'angles': [-3.2, 19.1, 90, 25.3, -0.6, -76.2],
                'wait_time': 3.0  # 等待3秒
            },
            {
                'name': '移动到初始位置',
                'angles': [-2.1, -90, 90, 90, -0.8, -76.4],
                'wait_time': 3.0  # 等待3秒
            }
        ]
    
    def move_to_angles(self, angles, description=""):
        """移动到指定角度
        
        Args:
            angles (list): 关节角度列表
            description (str): 动作描述
        """
        print(f"\n{description}")
        print(f"目标角度: {angles}")
        
        # 检查角度范围
        for i, angle in enumerate(angles):
            if angle < -90 or angle > 90:
                print(f"警告：关节{i+1}角度 {angle}° 超出范围 [-90°, 90°]")
        
        # 发送角度到舵机
        if self.servo_controller.set_all_servo_angles(angles):
            print(f"✓ {description} - 角度设置成功")
            return True
        else:
            print(f"✗ {description} - 角度设置失败")
            return False
    
    def execute_sequence(self):
        """执行完整的动作序列"""
        print("\n=== 开始执行机械臂动作序列 ===")
        
        for i, step in enumerate(self.sequence, 1):
            print(f"\n--- 步骤 {i}: {step['name']} ---")
            
            success = False
            
            if 'angles' in step:
                # 关节角度移动
                success = self.move_to_angles(step['angles'], step['name'])
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
        
        print("\n=== 动作序列执行完成 ===")
    
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
        # 创建序列控制器
        controller = RobotSequenceController(port=STUDENT_PORT)
        
        # 执行动作序列
        controller.execute_sequence()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行失败: {e}")
        print("请检查：")
        print("1. 串口连接是否正确")
        print("2. 舵机是否正常供电")
        print("3. 角度值是否在有效范围内")
    finally:
        try:
            controller.cleanup()
        except:
            pass

if __name__ == '__main__':
    main()