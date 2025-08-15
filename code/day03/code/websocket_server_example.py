#!/usr/bin/env python3
"""
WebSocket服务器示例
用于向机械臂模拟器发送舵机角度数据

使用方法:
1. 安装依赖: pip install websockets
2. 运行服务器: python websocket_server_example.py
3. 在浏览器中打开机械臂模拟器
4. 服务器会自动发送测试角度数据
"""

import asyncio
import websockets
import json
import math
import time

# 服务器配置
HOST = 'localhost'
PORT = 8080

# 连接的客户端列表
connected_clients = set()

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

async def send_test_angles():
    """发送测试角度数据"""
    global connected_clients
    start_time = time.time()
    
    while True:
        if connected_clients:
            # 生成测试角度数据（正弦波运动）
            current_time = time.time() - start_time
            
            # 为6个关节生成不同频率的正弦波角度（-90度到90度，即-1.57到1.57弧度）
            angles = [
                math.sin(current_time * 0.5) * 1.57,     # Rotation: 基座旋转
                math.sin(current_time * 0.3) * 1.57,     # Rotation2: 肩部
                math.sin(current_time * 0.4) * 1.57,     # Rotation3: 肘部
                math.sin(current_time * 0.6) * 1.57,     # Rotation4: 腕部1
                math.sin(current_time * 0.7) * 1.57,     # Rotation5: 腕部2
                math.sin(current_time * 0.8) * 0.785     # Rotation6: 腕部3 (0到90度)
            ]
            
            # 创建消息数据
            message_data = {
                "angles": angles,
                "timestamp": current_time
            }
            
            # 发送给所有连接的客户端
            message = json.dumps(message_data)
            print(f"发送角度数据: {[f'{angle:.3f}' for angle in angles]}")
            print(f"JSON消息: {message}")
            disconnected_clients = set()
            
            for client in connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # 移除断开的客户端
            connected_clients -= disconnected_clients
            
            print(f"发送角度数据: {[f'{angle:.3f}' for angle in angles]}")
        
        # 每100ms发送一次数据
        await asyncio.sleep(0.1)

async def send_manual_angles(angles):
    """手动发送指定的角度数据"""
    if connected_clients:
        message_data = {
            "angles": angles,
            "timestamp": time.time()
        }
        
        message = json.dumps(message_data)
        disconnected_clients = set()
        
        for client in connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        connected_clients -= disconnected_clients
        print(f"手动发送角度数据: {[f'{angle:.3f}' for angle in angles]}")

async def main():
    """主函数"""
    print(f"启动WebSocket服务器: ws://{HOST}:{PORT}")
    print("等待机械臂模拟器连接...")
    print("按 Ctrl+C 停止服务器")
    
    # 启动WebSocket服务器
    server = await websockets.serve(handle_client, HOST, PORT)
    
    # 启动测试数据发送任务
    send_task = asyncio.create_task(send_test_angles())
    
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
    # 示例：发送不同格式的数据
    print("WebSocket服务器示例")
    print("支持的数据格式:")
    print("1. 数组格式: {\"angles\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}")
    print("2. 对象格式: {\"servo1\": 0.1, \"servo2\": 0.2, ...}")
    print("3. 数字键格式: {\"1\": 0.1, \"2\": 0.2, ...}")
    print()
    
    start_server()