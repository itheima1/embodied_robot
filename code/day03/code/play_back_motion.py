#!/usr/bin/env python3
"""
WebSocket服务器 - 动作序列播放
从JSON文件读取动作序列并发送给机械臂模拟器

使用方法:
1. 安装依赖: pip install websockets
2. 运行服务器: python play_back_motion.py
3. 在浏览器中打开机械臂模拟器
4. 服务器会按时间序列播放动作数据
"""

import asyncio
import websockets
import json
import time
import os
import sys
from typing import List, Dict, Any

# 服务器配置
HOST = 'localhost'
PORT = 8080

# 连接的客户端列表
connected_clients = set()

# 动作数据
motion_data = None
motion_metadata = None
is_playing = False
play_start_time = 0
current_frame_index = 0

class MotionPlayer:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.motion_data = []
        self.metadata = {}
        self.is_loaded = False
        
    def load_motion_data(self) -> bool:
        """加载动作序列数据"""
        try:
            if not os.path.exists(self.json_file_path):
                print(f"错误: 文件不存在 {self.json_file_path}")
                return False
                
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.metadata = data.get('metadata', {})
            self.motion_data = data.get('motion_data', [])
            
            if not self.motion_data:
                print("错误: 动作数据为空")
                return False
                
            self.is_loaded = True
            print(f"成功加载动作数据:")
            print(f"  总帧数: {self.metadata.get('total_frames', len(self.motion_data))}")
            print(f"  持续时间: {self.metadata.get('duration', 0):.2f}秒")
            print(f"  关节ID: {self.metadata.get('joint_ids', [1,2,3,4,5,6])}")
            print(f"  录制时间: {self.metadata.get('recorded_at', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"加载动作数据失败: {e}")
            return False
    
    def get_frame_at_time(self, elapsed_time: float) -> Dict[str, Any]:
        """根据经过的时间获取对应的帧数据"""
        if not self.is_loaded or not self.motion_data:
            return None
            
        # 找到最接近当前时间的帧
        best_frame = self.motion_data[0]
        for frame in self.motion_data:
            if frame['timestamp'] <= elapsed_time:
                best_frame = frame
            else:
                break
                
        return best_frame
    
    def convert_angles_to_radians(self, angles_dict: Dict[str, float]) -> List[float]:
        """将角度字典转换为弧度列表"""
        # 按关节ID顺序排列角度
        joint_ids = self.metadata.get('joint_ids', [1, 2, 3, 4, 5, 6])
        angles_rad = []
        
        for joint_id in joint_ids:
            angle_deg = angles_dict.get(str(joint_id), 0.0)
            # 转换为弧度，注意这里的角度已经是度数，需要转换为弧度
            angle_rad = -angle_deg * 3.14159265359 / 180.0
            angles_rad.append(angle_rad)
            
        return angles_rad
    
    def get_total_duration(self) -> float:
        """获取动作总时长"""
        if not self.is_loaded:
            return 0.0
        return self.metadata.get('duration', 0.0)

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
                
                # 处理播放控制命令
                if data.get('command') == 'play':
                    await start_playback()
                elif data.get('command') == 'stop':
                    await stop_playback()
                elif data.get('command') == 'reset':
                    await reset_playback()
                    
            except json.JSONDecodeError:
                print(f"收到无效JSON消息: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"客户端断开连接: {websocket.remote_address}")

async def start_playback():
    """开始播放"""
    global is_playing, play_start_time, current_frame_index
    if not is_playing:
        is_playing = True
        play_start_time = time.time()
        current_frame_index = 0
        print("开始播放动作序列")

async def stop_playback():
    """停止播放"""
    global is_playing
    is_playing = False
    print("停止播放动作序列")

async def reset_playback():
    """重置播放"""
    global is_playing, play_start_time, current_frame_index
    is_playing = False
    play_start_time = 0
    current_frame_index = 0
    print("重置播放状态")

async def send_motion_angles(motion_player: MotionPlayer):
    """发送动作序列角度数据"""
    global connected_clients, is_playing, play_start_time
    
    while True:
        if connected_clients and motion_player.is_loaded:
            try:
                if is_playing:
                    # 计算经过的时间
                    elapsed_time = time.time() - play_start_time
                    total_duration = motion_player.get_total_duration()
                    
                    # 检查是否播放完成
                    if elapsed_time >= total_duration:
                        print("动作序列播放完成")
                        is_playing = False
                        continue
                    
                    # 获取当前时间对应的帧数据
                    current_frame = motion_player.get_frame_at_time(elapsed_time)
                    
                    if current_frame:
                        # 转换角度为弧度
                        angles_rad = motion_player.convert_angles_to_radians(current_frame['angles'])
                        
                        # 创建消息数据
                        message_data = {
                            "angles": angles_rad,
                            "timestamp": time.time(),
                            "frame_time": current_frame['timestamp'],
                            "elapsed_time": elapsed_time,
                            "progress": elapsed_time / total_duration if total_duration > 0 else 0
                        }
                        
                        # 发送给所有连接的客户端
                        message = json.dumps(message_data)
                        print(f"播放进度: {elapsed_time:.2f}/{total_duration:.2f}s ({message_data['progress']*100:.1f}%) - 角度: {[f'{angle:.3f}' for angle in angles_rad]}")
                        
                        disconnected_clients = set()
                        
                        for client in connected_clients:
                            try:
                                await client.send(message)
                            except websockets.exceptions.ConnectionClosed:
                                disconnected_clients.add(client)
                        
                        # 移除断开的客户端
                        connected_clients -= disconnected_clients
                else:
                    # 如果没有播放，发送初始位置
                    if motion_player.motion_data:
                        first_frame = motion_player.motion_data[0]
                        angles_rad = motion_player.convert_angles_to_radians(first_frame['angles'])
                        
                        message_data = {
                            "angles": angles_rad,
                            "timestamp": time.time(),
                            "frame_time": 0,
                            "elapsed_time": 0,
                            "progress": 0
                        }
                        
                        message = json.dumps(message_data)
                        
                        disconnected_clients = set()
                        
                        for client in connected_clients:
                            try:
                                await client.send(message)
                            except websockets.exceptions.ConnectionClosed:
                                disconnected_clients.add(client)
                        
                        connected_clients -= disconnected_clients
                        
            except Exception as e:
                print(f"发送动作角度时发生错误: {e}")
        
        # 每50ms发送一次数据（20fps）
        await asyncio.sleep(0.05)

async def main():
    """主函数"""
    # 检查命令行参数
    json_file = "arm_motion_20250725_165838.json"  # 默认文件
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    # 如果文件路径不是绝对路径，则相对于当前脚本目录
    if not os.path.isabs(json_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(script_dir, json_file)
    
    print(f"启动WebSocket服务器: ws://{HOST}:{PORT}")
    print(f"加载动作文件: {json_file}")
    
    # 初始化动作播放器
    motion_player = MotionPlayer(json_file)
    
    if not motion_player.load_motion_data():
        print("无法加载动作数据，退出程序")
        return
    
    print("等待机械臂模拟器连接...")
    print("控制命令:")
    print("  发送 {'command': 'play'} 开始播放")
    print("  发送 {'command': 'stop'} 停止播放")
    print("  发送 {'command': 'reset'} 重置播放")
    print("按 Ctrl+C 停止服务器")
    
    # 启动WebSocket服务器
    server = await websockets.serve(handle_client, HOST, PORT)
    
    # 启动动作数据发送任务
    send_task = asyncio.create_task(send_motion_angles(motion_player))
    
    # 自动开始播放
    await asyncio.sleep(1)  # 等待1秒让客户端连接
    await start_playback()
    
    try:
        # 同时运行服务器和发送任务
        await asyncio.gather(
            server.wait_closed(),
            send_task
        )
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
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
    print("WebSocket服务器 - 动作序列播放")
    print("功能: 从JSON文件读取动作序列并发送给机械臂模拟器")
    print(f"默认动作文件: arm_motion_20250725_165838.json")
    print("用法: python play_back_motion.py [动作文件.json]")
    print()
    
    start_server()