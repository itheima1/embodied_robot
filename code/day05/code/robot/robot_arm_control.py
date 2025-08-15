#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机械臂反解控制程序
结合反解求解器和舵机控制，实现TCP末端的六个方向移动控制

控制说明：
- W/S: 前进/后退 (X轴)
- A/D: 左移/右移 (Y轴) 
- Q/E: 上升/下降 (Z轴)
- R: 重置到初始位置
- ESC: 退出程序
"""

import serial
import time
import sys
import numpy as np
import cv2
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
    
    def get_servo_angle(self, servo_id):
        """读取舵机当前角度"""
        if not self._send_packet(servo_id, self.INST_READ, [self.ADDR_PRESENT_POSITION, 2]):
            return None

        result, error, data = self._read_packet()

        if result != self.COMM_SUCCESS or error != 0:
            return None
        
        if data and len(data) >= 2:
            position = data[0] | (data[1] << 8)
            angle = ((position - 1024.0) / (3072.0 - 1024.0)) * 180.0 - 90.0
            angle = max(-90.0, min(90.0, angle))
            return angle
        
        return None
    
    def get_all_servo_angles(self):
        """读取所有舵机当前角度"""
        angles = []
        for servo_id in range(1, 6):  # 读取前5个关节
            angle = self.get_servo_angle(servo_id)
            if angle is not None:
                angles.append(angle)
            else:
                print(f"读取舵机 {servo_id} 角度失败")
                angles.append(0.0)  # 失败时使用默认值
            time.sleep(0.01)  # 小延时避免通信冲突
        return angles
    
    def set_all_servo_angles(self, angles):
        """设置所有舵机角度"""
        success = True
        for i, angle in enumerate(angles[:5]):  # 只使用前5个关节
            if not self.set_servo_angle(i + 1, angle):
                success = False
                print(f"设置舵机 {i+1} 角度失败")
            time.sleep(0.01)  # 小延时避免通信冲突
        return success

class RobotArmController:
    """机械臂控制器主类"""
    
    def __init__(self, port="COM6"):
        # 初始化反解求解器
        self.solver = InverseKinematicsSolver()
        
        # 初始化舵机控制器
        self.servo_controller = ServoController(port=port)
        
        # 启用所有舵机扭矩
        print("正在启用舵机扭矩...")
        for servo_id in range(1, 6):
            self.servo_controller.enable_torque(servo_id)
            time.sleep(0.05)
        print("舵机扭矩已启用")
        
        # 读取机械臂当前角度作为初始角度
        self.read_initial_angles()
        
        # 创建显示窗口
        self.create_display_window()
    
    def read_initial_angles(self):
        """读取机械臂当前角度作为初始角度"""
        print("正在读取机械臂当前角度...")
        current_angles_deg = self.servo_controller.get_all_servo_angles()
        self.current_angles = np.deg2rad(current_angles_deg)
        print(f"当前角度 (度): {current_angles_deg}")
        print(f"当前角度 (弧度): {self.current_angles}")
        
        # 验证读取的角度是否合理
        if all(angle is not None for angle in current_angles_deg):
            print("成功读取机械臂当前角度")
        else:
            print("警告：部分角度读取失败，使用默认值")
            self.current_angles = np.deg2rad([0, 30, -30, -30, 0])
    
    def move_to_initial_position(self):
        """移动到初始位置（现在用于重置功能）"""
        print("重置到默认位置...")
        default_angles = [0, 30, -30, -30, 0]
        self.current_angles = np.deg2rad(default_angles)
        if self.servo_controller.set_all_servo_angles(default_angles):
            print(f"重置位置设置完成: {default_angles}")
        else:
            print("重置位置设置失败")
        time.sleep(1.0)  # 等待移动完成
    
    def create_display_window(self):
        """创建显示窗口"""
        # 创建一个黑色背景的显示窗口
        self.display_img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.namedWindow('Robot Arm Control', cv2.WINDOW_AUTOSIZE)
        self.update_display()
    
    def update_display(self):
        """更新显示内容"""
        # 清空显示
        self.display_img.fill(0)
        
        # 获取当前TCP位置
        tcp_pos = self.solver.get_current_tcp_position(self.current_angles)
        angles_deg = np.rad2deg(self.current_angles)
        
        # 显示信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        # 标题
        cv2.putText(self.display_img, 'Robot Arm Control System', (200, 50), 
                   font, 1.0, (0, 255, 255), thickness)
        
        # TCP位置信息
        cv2.putText(self.display_img, 'TCP Position (m):', (50, 120), 
                   font, font_scale, color, thickness)
        cv2.putText(self.display_img, f'X: {tcp_pos[0]:.3f}', (50, 150), 
                   font, font_scale, (0, 255, 0), thickness)
        cv2.putText(self.display_img, f'Y: {tcp_pos[1]:.3f}', (50, 180), 
                   font, font_scale, (0, 255, 0), thickness)
        cv2.putText(self.display_img, f'Z: {tcp_pos[2]:.3f}', (50, 210), 
                   font, font_scale, (0, 255, 0), thickness)
        
        # 关节角度信息
        cv2.putText(self.display_img, 'Joint Angles (deg):', (400, 120), 
                   font, font_scale, color, thickness)
        for i, angle in enumerate(angles_deg):
            cv2.putText(self.display_img, f'Joint {i+1}: {angle:.1f}', 
                       (400, 150 + i*30), font, font_scale, (255, 255, 0), thickness)
        
        # 控制说明
        cv2.putText(self.display_img, 'Controls:', (50, 320), 
                   font, font_scale, (255, 0, 255), thickness)
        controls = [
            'W/S: Forward/Backward (X-axis)',
            'A/D: Left/Right (Y-axis)',
            'Q/E: Up/Down (Z-axis)',
            'R: Reset to initial position',
            'ESC: Exit program'
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(self.display_img, control, (50, 350 + i*30), 
                       font, 0.6, (200, 200, 200), 1)
        
        # 显示图像
        cv2.imshow('Robot Arm Control', self.display_img)
    
    def execute_movement(self, direction_name, move_function):
        """执行移动并更新显示"""
        success, new_angles, error = move_function(self.current_angles)
        
        if success:
            # 更新关节角度
            self.current_angles = new_angles
            
            # 发送到舵机
            angles_deg = np.rad2deg(new_angles)
            if self.servo_controller.set_all_servo_angles(angles_deg):
                print(f"\r{direction_name}移动成功 - TCP位置: {self.solver.get_current_tcp_position(new_angles)}", end="")
            else:
                print(f"\r{direction_name}移动计算成功，但舵机控制失败", end="")
        else:
            print(f"\r{direction_name}移动失败: {error}", end="")
        
        # 更新显示
        self.update_display()
        return success
    
    def run(self):
        """主运行循环"""
        print("\n=== 机械臂反解控制系统启动 ===")
        print("使用WASD QE控制TCP移动，R重置，ESC退出")
        
        try:
            while True:
                key = cv2.waitKey(30) & 0xFF
                
                if key == 27:  # ESC键
                    break
                elif key == ord('w') or key == ord('W'):
                    self.execute_movement("前进", self.solver.move_forward)
                elif key == ord('s') or key == ord('S'):
                    self.execute_movement("后退", self.solver.move_backward)
                elif key == ord('a') or key == ord('A'):
                    self.execute_movement("左移", self.solver.move_left)
                elif key == ord('d') or key == ord('D'):
                    self.execute_movement("右移", self.solver.move_right)
                elif key == ord('q') or key == ord('Q'):
                    self.execute_movement("上升", self.solver.move_up)
                elif key == ord('e') or key == ord('E'):
                    self.execute_movement("下降", self.solver.move_down)
                elif key == ord('r') or key == ord('R'):
                    print("\n重置到初始位置...")
                    self.current_angles = np.deg2rad([0, 30, -30, -30, 0])
                    self.move_to_initial_position()
                    self.update_display()
                
                # 检查窗口是否被关闭
                if cv2.getWindowProperty('Robot Arm Control', cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"\n发生错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        cv2.destroyAllWindows()
        self.servo_controller.close()
        print("程序结束")

def main():
    """主函数"""
    # 配置参数
    STUDENT_PORT = "COM6"  # 根据实际情况调整串口
    
    try:
        # 创建机械臂控制器
        controller = RobotArmController(port=STUDENT_PORT)
        
        # 运行控制程序
        controller.run()
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        print("请检查：")
        print("1. 串口连接是否正确")
        print("2. 舵机是否正常供电")
        print("3. inverse_kinematics_solver.py 文件是否存在")

if __name__ == '__main__':
    main()