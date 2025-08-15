#!/usr/bin/env python3
"""
WebSocket服务器 - 键盘控制机械臂逆运动学
结合键盘控制和WebSocket服务器，让用户通过按键控制模拟器中的机械臂

使用方法:
1. 安装依赖: pip install websockets keyboard numpy scipy
2. 运行服务器: python invers_arm_server.py
3. 在浏览器中打开机械臂模拟器
4. 使用键盘控制机械臂移动:
   - W: 向前移动
   - S: 向后移动
   - A: 向左移动
   - D: 向右移动
   - Q: 向上移动
   - E: 向下移动
   - R: 重置到初始位置
   - ESC: 退出程序
"""

import asyncio
import websockets
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import keyboard
import threading
from typing import Optional, List

# 服务器配置
HOST = 'localhost'
PORT = 8080

# 连接的客户端列表
connected_clients = set()

# =============================================================================
# 1. 机器人模型 (与keyboard_control.py相同)
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

# 关节限位
JOINT_LIMIT_MIN = np.deg2rad(np.array([-90.0] * 5))
JOINT_LIMIT_MAX = np.deg2rad(np.array([90.0] * 5))
GRIPPER_LIMIT_MIN = 0.0
GRIPPER_LIMIT_MAX = np.deg2rad(90.0)

def get_flange_to_tcp_transform():
    T_tcp = np.identity(4)
    T_tcp[:3, :3] = R.from_euler('xyz', TCP_OFFSET_RPY).as_matrix()
    T_tcp[:3, 3] = TCP_OFFSET_XYZ
    return T_tcp

T_FLANGE_TO_TCP = get_flange_to_tcp_transform()

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
# 2. 机器人控制器类
# =============================================================================
class RobotController:
    def __init__(self, initial_q_all):
        self.current_q = np.array(initial_q_all, dtype=float)
        self.running = True
        self.movement_lock = threading.Lock()
        print("机器人控制器已初始化。")
        self.print_state()

    def get_joint_angles_5dof(self):
        return self.current_q[:5]

    def set_joint_angles_5dof(self, new_q_5dof):
        if np.any(new_q_5dof < JOINT_LIMIT_MIN) or np.any(new_q_5dof > JOINT_LIMIT_MAX):
            return False
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
        print(f"\n当前关节角度(度): [{angles_deg[0]:.1f}, {angles_deg[1]:.1f}, {angles_deg[2]:.1f}, {angles_deg[3]:.1f}, {angles_deg[4]:.1f}], 夹爪: {gripper_deg:.1f}")
        print(f"当前TCP位置(米):  [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    def calculate_movement(self, direction, step_size_mm=5.0, damping_factor=0.05):
        """计算指定方向移动后的关节角度"""
        direction = np.array(direction) / np.linalg.norm(direction)
        delta_X = direction * (step_size_mm / 1000.0)
        
        q_5dof = self.get_joint_angles_5dof()
        Jp = get_position_jacobian(q_5dof)
        
        # 使用阻尼最小二乘法计算关节角度变化
        Jp_T = Jp.T
        lambda_sq = (damping_factor**2) * np.identity(3)
        try:
            inv_term = np.linalg.inv(Jp @ Jp_T + lambda_sq)
            delta_q = Jp_T @ inv_term @ delta_X
        except np.linalg.LinAlgError:
            print("错误: 无法计算逆运动学解")
            return None
        
        new_q = q_5dof + delta_q
        
        # 检查关节限位
        if np.any(new_q < JOINT_LIMIT_MIN) or np.any(new_q > JOINT_LIMIT_MAX):
            print("警告: 目标角度超出关节限位！")
            return None
            
        return new_q

    def execute_movement(self, direction_name, direction_vector):
        """执行移动并返回是否成功"""
        with self.movement_lock:
            print(f"\n=== {direction_name} ===")
            print("移动前状态:")
            self.print_state()
            
            new_angles = self.calculate_movement(direction_vector)
            if new_angles is not None:
                print(f"\n移动后关节角度(度): [{np.rad2deg(new_angles[0]):.1f}, {np.rad2deg(new_angles[1]):.1f}, {np.rad2deg(new_angles[2]):.1f}, {np.rad2deg(new_angles[3]):.1f}, {np.rad2deg(new_angles[4]):.1f}]")
                
                # 计算移动后的TCP位置
                _, T_final_tcp = forward_kinematics(new_angles)
                new_pos = T_final_tcp[:3, 3]
                print(f"移动后TCP位置(米):  [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]")
                
                # 实际执行移动
                if self.set_joint_angles_5dof(new_angles):
                    print("移动执行成功！")
                    return True
                else:
                    print("移动执行失败！")
                    return False
            else:
                print("无法执行此移动")
                return False

    def reset_to_initial(self):
        """重置到初始位置"""
        with self.movement_lock:
            print("\n=== 重置到初始位置 ===")
            self.current_q = np.array([0.0, np.deg2rad(30), np.deg2rad(-30), np.deg2rad(-30), 0.0, 0.0])
            self.print_state()
            return True

    def get_current_angles(self):
        """获取当前所有关节角度"""
        return self.current_q.copy()

# =============================================================================
# 3. WebSocket服务器相关函数
# =============================================================================

# 全局机器人控制器
robot_controller = None

async def handle_client(websocket):
    """处理客户端连接"""
    print(f"新客户端连接: {websocket.remote_address}")
    connected_clients.add(websocket)
    
    try:
        # 保持连接并处理消息
        async for message in websocket:
            try:
                # 可以在这里处理客户端发送的消息
                data = json.loads(message)
                print(f"收到客户端消息: {data}")
            except json.JSONDecodeError:
                print(f"收到无效JSON消息: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"客户端断开连接: {websocket.remote_address}")

async def send_robot_angles():
    """发送机器人角度数据"""
    global connected_clients, robot_controller
    
    while robot_controller and robot_controller.running:
        if connected_clients and robot_controller:
            try:
                # 获取当前所有关节角度
                angles = robot_controller.get_current_angles().tolist()
                
                # 创建消息数据
                message_data = {
                    "angles": angles,
                    "timestamp": time.time()
                }
                
                # 发送给所有连接的客户端
                message = json.dumps(message_data)
                print(f"发送角度数据: {[f'{angle:.3f}' for angle in angles]}")
                
                disconnected_clients = set()
                
                for client in connected_clients:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                
                # 移除断开的客户端
                connected_clients -= disconnected_clients
                
            except Exception as e:
                print(f"发送角度数据时发生错误: {e}")
        
        # 每100ms发送一次数据
        await asyncio.sleep(0.1)

# =============================================================================
# 4. 键盘控制逻辑
# =============================================================================
def keyboard_control_thread():
    """键盘控制线程"""
    global robot_controller
    
    print("\n=== 键盘控制说明 ===")
    print("W - 向前移动")
    print("S - 向后移动")
    print("A - 向左移动")
    print("D - 向右移动")
    print("Q - 向上移动")
    print("E - 向下移动")
    print("R - 重置到初始位置")
    print("ESC - 退出程序")
    print("===================\n")
    
    # 定义移动方向映射
    direction_map = {
        'w': ('向前移动', [1, 0, 0]),
        's': ('向后移动', [-1, 0, 0]),
        'a': ('向左移动', [0, 1, 0]),
        'd': ('向右移动', [0, -1, 0]),
        'q': ('向上移动', [0, 0, 1]),
        'e': ('向下移动', [0, 0, -1])
    }
    
    while robot_controller and robot_controller.running:
        try:
            # 等待键盘输入
            print("\n请按键控制机械臂 (按ESC退出):")
            event = keyboard.read_event()
            
            if event.event_type == keyboard.KEY_DOWN:
                key = event.name.lower()
                
                if key == 'esc':
                    print("退出程序...")
                    robot_controller.running = False
                    break
                elif key == 'r':
                    robot_controller.reset_to_initial()
                elif key in direction_map:
                    direction_name, direction_vector = direction_map[key]
                    robot_controller.execute_movement(direction_name, direction_vector)
                else:
                    print(f"未识别的按键: {key}")
                    
        except KeyboardInterrupt:
            print("\n程序被中断")
            robot_controller.running = False
            break
        except Exception as e:
            print(f"发生错误: {e}")

# =============================================================================
# 5. 主程序
# =============================================================================
async def main():
    """主函数"""
    global robot_controller
    
    print(f"启动WebSocket服务器: ws://{HOST}:{PORT}")
    print("等待机械臂模拟器连接...")
    print("按 Ctrl+C 停止服务器")
    
    # 初始化机器人控制器到一个合适的工作姿态
    initial_angles = np.array([0.0, np.deg2rad(30), np.deg2rad(-30), np.deg2rad(-30), 0.0, 0.0])
    robot_controller = RobotController(initial_angles)
    
    print("\n机械臂键盘控制程序启动")
    print("初始工作姿态:")
    robot_controller.print_state()
    
    # 启动键盘控制线程
    keyboard_thread = threading.Thread(target=keyboard_control_thread, daemon=True)
    keyboard_thread.start()
    
    # 启动WebSocket服务器
    server = await websockets.serve(handle_client, HOST, PORT)
    
    # 启动角度数据发送任务
    send_task = asyncio.create_task(send_robot_angles())
    
    try:
        # 同时运行服务器和发送任务
        await asyncio.gather(
            server.wait_closed(),
            send_task
        )
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
        if robot_controller:
            robot_controller.running = False
        server.close()
        await server.wait_closed()
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        
        print("服务器已停止")

def start_server():
    """启动WebSocket服务器"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止")

if __name__ == "__main__":
    print("WebSocket服务器 - 键盘控制机械臂逆运动学")
    print("功能: 通过键盘控制模拟器中的机械臂移动")
    print(f"服务器地址: ws://{HOST}:{PORT}")
    print("支持的按键操作:")
    print("  W/S: 前进/后退")
    print("  A/D: 左移/右移")
    print("  Q/E: 上升/下降")
    print("  R: 重置位置")
    print("  ESC: 退出")
    print()
    
    start_server()