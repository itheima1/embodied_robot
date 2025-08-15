#!/usr/bin/env python3
"""
WebSocket服务器 - 真实舵机角度数据
从真实舵机读取角度数据并发送给机械臂模拟器

使用方法:
1. 安装依赖: pip install websockets pyserial
2. 确保舵机连接到正确的串口
3. 运行服务器: python websocket_server.py
4. 在浏览器中打开机械臂模拟器
5. 服务器会实时发送真实舵机角度数据
"""

import asyncio
import websockets
import json
import time
import struct
import enum
import logging
from serial import Serial
from serial.tools import list_ports

# 服务器配置
HOST = 'localhost'
PORT = 8080

# 串口配置
SERIAL_PORT = 'COM6'  # 根据实际情况修改
BAUDRATE = 230400

# 连接的客户端列表
connected_clients = set()

# region: 舵机通信相关类和函数（从02_read_all_teacher_angles.py复制）

def bytes_to_short(data: bytes, signed: bool = False, byteorder: str = 'little') -> int:
    if len(data) != 2:
        raise ValueError("Data must be exactly 2 bytes long")
    prefix = '<' if byteorder == 'little' else '>'
    format_char = 'h' if signed else 'H'
    return struct.unpack(f"{prefix}{format_char}", data)[0]

def short_to_bytes(value: int, signed: bool = False, byteorder: str = 'little') -> bytes:
    if signed:
        format_char = 'h'
        min_val, max_val = -32768, 32767
    else:
        format_char = 'H'
        min_val, max_val = 0, 65535
    if not (min_val <= value <= max_val):
        raise OverflowError(f"Value {value} out of range for {'signed' if signed else 'unsigned'} short")
    prefix = '<' if byteorder == 'little' else '>'
    return struct.pack(f"{prefix}{format_char}", value)

class Address(enum.Enum):
    DEVICE_UUID         = (0, 4)
    VERSION             = (4, 2)
    MOTOR_TYPE          = (6, 1)
    CURRENT_POSITION    = (7, 2)
    CURRENT_SPEED       = (9, 2)
    CURRENT_LOAD        = (11, 2)
    CURRENT_VOLTAGE     = (13, 1)
    CURRENT_CURRENT     = (14, 2)
    CURRENT_TEMPERATURE = (16, 1)
    TORQUE_ENABLE       = (50, 1)
    TARGET_POSITION     = (51, 2)
    ID                  = (70, 1)
    MIN_POSITION        = (71, 2)
    MAX_POSITION        = (73, 2)
    POSITION_OFFSET     = (75, 2)
    MAX_VOLTAGE         = (77, 1)
    MIN_VOLTAGE         = (78, 1)
    MAX_TEMPERATURE     = (79, 1)
    MAX_CURRENT         = (80, 2)
    KP                  = (82, 1)
    KI                  = (83, 1)
    KD                  = (84, 1)

    @classmethod
    def get_address(cls, address:int):
        for addr in cls:
            if addr.value[0] == address:
                return addr
        return None

class ErrorCode(enum.Enum):
    SUCCESS             = 0
    WRITE_ERROR         = 1
    READ_ERROR          = 2
    READ_TIMEOUT        = 3

class Result:
    def __init__(self, error: ErrorCode = ErrorCode.SUCCESS, frame: list[int] = None, input = None):
        self.__error_code = error
        self.__frame = frame
        self.__input = input
        self.__value_map = {}

        if frame is None or input is None:
            return

        id = frame[2]
        cmd = frame[3]
        if cmd != 0x03:
            return
        if id != 0xFF and id < 128 and id >= 248:
            return

        addresses = []
        if isinstance(input, Address):
            addresses.append(input)
        elif isinstance(input, list):
            addresses.extend(input)

        cnt = 6 if id == 0xFF else 5

        while cnt < len(frame) - 2:
            addr = Address.get_address(frame[cnt])
            if addr is None:
                break
            addr_int = addr.value[0]
            addr_len = addr.value[1]

            if addr_len == 1:
                self.__value_map[addr_int] = frame[cnt+1]
            elif addr_len == 2:
                self.__value_map[addr_int] = bytes_to_short(bytearray(frame[cnt+1:cnt+3]))
            elif addr_len == 4:
                self.__value_map[addr_int] = bytes_to_int(bytearray(frame[cnt+1:cnt+5]))
            cnt += addr_len + 1

    def is_success(self) -> bool:
        return self.__error_code == ErrorCode.SUCCESS

    def get_error_code(self) -> int:
        return self.__error_code.value

    def get_data(self, address: Address) -> int:
        address_int = address.value[0]
        return self.__value_map.get(address_int)

class PortHandler:
    def __init__(self):
        self.__serial: Serial = None
        self._port = None
        self._baudrate = BAUDRATE
        self._bytesize = 8
        self._parity = 'N'
        self._stopbits = 1
        self._read_timeout = None
        self._write_timeout = None
        self.__is_running = False

    def open(self, port) -> bool:
        self.close()
        try:
            self._port = port
            self.__serial = Serial(port=port, baudrate=self._baudrate, bytesize=self._bytesize, 
                                 parity=self._parity, stopbits=self._stopbits, 
                                 timeout=self._read_timeout, write_timeout=self._write_timeout)
            self.__is_running = True
            return True
        except Exception as e:
            print(f"串口连接失败: {e}")
            return False

    def is_open(self) -> bool:
        return self.__serial and self.__serial.is_open

    def close(self):
        if self.__serial and self.__serial.is_open:
            self.__serial.close()
            self.__is_running = False
            self.__serial = None

    def read_port(self, length:int):
        if self.__serial and self.__serial.is_open:
            return self.__serial.read(length)

    def write_port(self, data):
        if self.__serial and self.__serial.is_open:
            self.__serial.reset_input_buffer()
            self.__serial.write(data)
            self.__serial.flush()

    def in_waiting(self):
        if self.__serial and self.__serial.is_open:
            return self.__serial.in_waiting
        return 0

FRAME_HEADER    = 0xAA
FRAME_TAIL      = 0xBB
FRAME_CMD_READ  = 0x03

def checksum(id: int, cmd: int, data: list[int]) -> int:
    return (id + cmd + len(data) + sum(data)) & 0xFF

def frame_generator(id: int, cmd: int, data: list[int]) -> bytearray:
    frame = bytearray()
    frame.append(FRAME_HEADER)
    frame.append(FRAME_HEADER)
    frame.append(id)
    frame.append(cmd)
    frame.append(len(data))
    for d in data:
        frame.append(d)
    frame.append(checksum(id, cmd, data))
    frame.append(FRAME_TAIL)
    return frame

class SyncConnector:
    def __init__(self, portHandler: PortHandler):
        self.__port_handler = portHandler

    def _parse_response_frame(self) -> Result:
        retry_cnt = 0
        read_list = []
        state = 0
        self.__port_handler._read_timeout = 1
        while True:
            in_waiting = self.__port_handler.in_waiting()
            if in_waiting == 0:
                if retry_cnt < 5:
                    retry_cnt += 1
                    time.sleep(0.01)
                    continue
                else:
                    state = -1
                    break
            read_list.extend(list(self.__port_handler.read_port(in_waiting)))
            while len(read_list) >= 7:
                if read_list[0] != FRAME_HEADER or read_list[1] != FRAME_HEADER:
                    read_list.pop(0)
                    continue
                data_length = read_list[4]
                if data_length > 48 or len(read_list) < 7 + data_length or read_list[6 + data_length] != FRAME_TAIL:
                    read_list.pop(0)
                    continue
                checksum_val = sum(read_list[2:5 + data_length]) & 0xFF
                if checksum_val != read_list[5 + data_length]:
                    read_list.pop(0)
                    continue
                read_list = read_list[0:7 + data_length]
                state = 1
                break
            if state == 1:
                break
        if state == -1:
            return Result(error=ErrorCode.read_TIMEOUT)
        return Result(frame=read_list, input=self.last_read_address)

    def read(self, id_list: list[int], address_list: list[Address]) -> Result:
        self.last_read_address = address_list
        data = []
        for address in address_list:
            data.extend([address.value[0], address.value[1]])
        frame = frame_generator(id_list[0], FRAME_CMD_READ, data)
        self.__port_handler.write_port(frame)
        return self._parse_response_frame()

class ServoAngleReader:
    def __init__(self, port: str):
        self.port = port
        self.__port_handler: PortHandler = PortHandler()
        self.__sync_connector: SyncConnector = SyncConnector(self.__port_handler)
        self.is_connected = False
        # 舵机参数配置
        self.homing_offset = 2048  # 零位偏移
        self.resolution = 4096     # 分辨率

    def connect(self):
        """连接到串口"""
        if self.is_connected:
            return True
        
        if not self.__port_handler.open(self.port):
            return False
        
        self.is_connected = True
        print(f"成功连接到串口 {self.port}")
        return True

    def disconnect(self):
        """断开串口连接"""
        if not self.is_connected:
            return
        
        self.__port_handler.close()
        self.is_connected = False
        print("已断开串口连接")

    def read_angle(self, motor_id: int) -> float:
        """读取指定ID舵机的角度
        
        Args:
            motor_id: 舵机ID
            
        Returns:
            float: 舵机角度（弧度），如果读取失败返回None
        """
        if not self.is_connected:
            return None
        
        try:
            # 读取当前位置
            result = self.__sync_connector.read([motor_id], [Address.CURRENT_POSITION])
            
            if result.is_success():
                raw_position = result.get_data(Address.CURRENT_POSITION)
                if raw_position is not None:
                    # 转换为角度（度）
                    angle_deg = ((raw_position - self.homing_offset) / self.resolution) * 360
                    
                    # 对第6号舵机（夹爪）进行角度映射：打开=0度，关闭=-90度
                    if motor_id == 6:
                        angle_deg = -angle_deg-90
                    
                    # 转换为弧度
                    angle_rad = -angle_deg * 3.14159265359 / 180.0
                    return angle_rad
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"读取舵机ID {motor_id} 时发生异常: {e}")
            return None

    def read_all_angles(self, motor_ids: list[int]) -> list[float]:
        """读取所有舵机的角度
        
        Args:
            motor_ids: 舵机ID列表
            
        Returns:
            list[float]: 角度列表（弧度），失败的舵机返回0.0
        """
        angles = []
        for motor_id in motor_ids:
            angle = self.read_angle(motor_id)
            if angle is not None:
                angles.append(angle)
            else:
                angles.append(0.0)  # 读取失败时使用默认值
        return angles

# endregion

# 全局舵机读取器
servo_reader = None

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

async def send_real_servo_angles():
    """发送真实舵机角度数据"""
    global connected_clients, servo_reader
    
    motor_ids = [1, 2, 3, 4, 5, 6]  # 舵机ID列表
    
    while True:
        if connected_clients and servo_reader and servo_reader.is_connected:
            try:
                # 读取所有舵机的角度
                angles = servo_reader.read_all_angles(motor_ids)
                
                # 创建消息数据
                message_data = {
                    "angles": angles,
                    "timestamp": time.time()
                }
                
                # 发送给所有连接的客户端
                message = json.dumps(message_data)
                print(f"发送真实角度数据: {[f'{angle:.3f}' for angle in angles]}")
                
                disconnected_clients = set()
                
                for client in connected_clients:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                
                # 移除断开的客户端
                connected_clients -= disconnected_clients
                
            except Exception as e:
                print(f"读取舵机角度时发生错误: {e}")
        
        # 每100ms发送一次数据
        await asyncio.sleep(0.1)

async def main():
    """主函数"""
    global servo_reader
    
    print(f"启动WebSocket服务器: ws://{HOST}:{PORT}")
    print(f"尝试连接串口: {SERIAL_PORT}")
    
    # 初始化舵机读取器
    servo_reader = ServoAngleReader(SERIAL_PORT)
    
    if not servo_reader.connect():
        print(f"无法连接到串口 {SERIAL_PORT}，请检查串口设置")
        print("可用串口列表:")
        for port in list_ports.comports():
            print(f"  {port.device} - {port.description}")
        return
    
    print("等待机械臂模拟器连接...")
    print("按 Ctrl+C 停止服务器")
    
    # 启动WebSocket服务器
    server = await websockets.serve(handle_client, HOST, PORT)
    
    # 启动真实角度数据发送任务
    send_task = asyncio.create_task(send_real_servo_angles())
    
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
        
        # 断开串口连接
        if servo_reader:
            servo_reader.disconnect()
        
        print("服务器已停止")

def start_server():
    """启动WebSocket服务器"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止")

if __name__ == "__main__":
    print("WebSocket服务器 - 真实舵机角度数据")
    print("功能: 从真实舵机读取角度并发送给机械臂模拟器")
    print(f"串口: {SERIAL_PORT}")
    print(f"波特率: {BAUDRATE}")
    print("舵机ID: 1-6")
    print()
    
    start_server()