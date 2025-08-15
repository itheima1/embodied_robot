#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遥操作控制程序
实现学生端舵机实时复制教师端舵机的动作
"""

import time
import struct
import enum
import sys
import threading
from copy import deepcopy
import numpy as np
from serial import Serial
from serial.tools import list_ports
import serial

# ===== 教师端读取相关代码 =====

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

def bytes_to_int(byte_data, signed=False, byteorder='little'):
    if len(byte_data) != 4:
        raise ValueError("输入必须是 4 字节")
    fmt_char = 'i' if signed else 'I'
    fmt_str = ('>' if byteorder == 'big' else '<') + fmt_char
    return struct.unpack(fmt_str, byte_data)[0]

def int_to_bytes(int_value, signed=False, byteorder='little'):
    if signed and not (-2**31 <= int_value < 2**31):
        raise ValueError("有符号整数超出 4 字节范围")
    elif not signed and not (0 <= int_value < 2**32):
        raise ValueError("无符号整数超出 4 字节范围")
    fmt_char = 'i' if signed else 'I'
    fmt_str = ('>' if byteorder == 'big' else '<') + fmt_char
    return struct.pack(fmt_str, int_value)

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
        self._baudrate = 230400
        self._bytesize = 8
        self._parity = 'N'
        self._stopbits = 1
        self._read_timeout = None
        self._write_timeout = None
        self.__is_running = False

    @property
    def baudrate(self):
        return self._baudrate

    @baudrate.setter
    def baudrate(self, value):
        if self.__serial and self.__serial.is_open:
            raise ValueError("无法修改已打开的串口波特率")
        self._baudrate = value

    def open(self, port) -> bool:
        self.close()
        try:
            self._port = port
            self.__serial = Serial(port=port, baudrate=self._baudrate, bytesize=self._bytesize, parity=self._parity, stopbits=self._stopbits, timeout=self._read_timeout, write_timeout=self._write_timeout)
            self.__is_running = True
            return True
        except Exception:
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
FRAME_CMD_READ          = 0x03

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
            return Result(error=ErrorCode.READ_TIMEOUT)
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
            print(f"已经连接到端口 {self.port}")
            return True
        
        if not self.__port_handler.open(self.port):
            print(f"无法连接到端口 {self.port}")
            return False
        
        self.is_connected = True
        print(f"成功连接到端口 {self.port}")
        return True

    def disconnect(self):
        """断开串口连接"""
        if not self.is_connected:
            print("未连接到任何端口")
            return
        
        self.__port_handler.close()
        self.is_connected = False
        print("已断开连接")

    def read_angle(self, motor_id: int) -> float:
        """读取指定ID舵机的角度
        
        Args:
            motor_id: 舵机ID
            
        Returns:
            float: 舵机角度（度），如果读取失败返回None
        """
        if not self.is_connected:
            return None
        
        try:
            # 读取当前位置
            result = self.__sync_connector.read([motor_id], [Address.CURRENT_POSITION])
            
            if result.is_success():
                raw_position = result.get_data(Address.CURRENT_POSITION)
                if raw_position is not None:
                    # 转换为角度
                    angle = ((raw_position - self.homing_offset) / self.resolution) * 360
                    return angle
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            return None

# ===== 学生端控制相关代码 =====

class ServoController:
    def __init__(self, port="COM6", baudrate=1000000, timeout=0.1):
        self.port_name = port
        self.serial_port = None
        # Constants
        self.ADDR_GOAL_POSITION = 42
        self.ADDR_TORQUE_ENABLE = 40
        self.INST_WRITE = 3
        self.COMM_SUCCESS = 0
        self.COMM_RX_TIMEOUT = -6
        self.COMM_RX_CORRUPT = -7

        try:
            self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            time.sleep(0.1)
            self.serial_port.reset_input_buffer()
            print(f"Successfully opened port {port}.")
        except serial.SerialException as e:
            print(f"Fatal: Could not open port {port}: {e}")
            raise e

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"Port {self.port_name} closed.")

    def _calculate_checksum(self, data):
        return (~sum(data)) & 0xFF

    def _send_packet(self, servo_id, instruction, parameters=None):
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

    def _write_register(self, servo_id, address, value, size=2):
        params = [address]
        if size == 1:
            params.append(value & 0xFF)
        elif size == 2:
            params.extend([value & 0xFF, (value >> 8) & 0xFF])
        else:
            return False
        
        if not self._send_packet(servo_id, self.INST_WRITE, params):
            return False
        
        return True

    def enable_torque(self, servo_id):
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, 1, size=1)

    def set_servo_angle(self, servo_id, angle):
        """Sets the servo to a specific angle (-90 to 90 degrees)."""
        # Map angle (-90 to 90) to position (1024 to 3072)
        position = int(((angle + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        # Clamp the value to be safe
        position = max(1024, min(3072, position))
        
        return self._write_register(servo_id, self.ADDR_GOAL_POSITION, position, size=2)

# ===== 遥操作主类 =====

class RemoteOperationController:
    def __init__(self, teacher_port="COM6", student_port="COM6", servo_ids=None):
        """
        遥操作控制器
        
        Args:
            teacher_port: 教师端串口
            student_port: 学生端串口
            servo_ids: 舵机ID列表，默认为[1,2,3,4,5,6]
        """
        if servo_ids is None:
            servo_ids = [1, 2, 3, 4, 5, 6]
        
        self.servo_ids = servo_ids
        self.teacher_port = teacher_port
        self.student_port = student_port
        
        # 初始化读取器和控制器
        self.teacher_reader = None
        self.student_controller = None
        
        # 控制标志
        self.is_running = False
        self.operation_thread = None
        
        # 角度缓存
        self.last_angles = {servo_id: 0.0 for servo_id in self.servo_ids}
        
    def connect(self):
        """连接教师端和学生端"""
        try:
            # 连接教师端
            print("正在连接教师端...")
            self.teacher_reader = ServoAngleReader(self.teacher_port)
            if not self.teacher_reader.connect():
                print("教师端连接失败")
                return False
            
            # 连接学生端
            print("正在连接学生端...")
            self.student_controller = ServoController(port=self.student_port, baudrate=1000000)
            
            # 启用学生端舵机扭矩
            print("启用学生端舵机扭矩...")
            for servo_id in self.servo_ids:
                self.student_controller.enable_torque(servo_id)
                time.sleep(0.05)
            
            print("连接成功！")
            return True
            
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.stop_operation()
        
        if self.teacher_reader:
            self.teacher_reader.disconnect()
            self.teacher_reader = None
            
        if self.student_controller:
            self.student_controller.close()
            self.student_controller = None
            
        print("已断开所有连接")
    
    def _operation_loop(self):
        """遥操作主循环"""
        print("开始遥操作，学生端将复制教师端动作...")
        
        while self.is_running:
            try:
                # 读取教师端所有舵机角度
                teacher_angles = {}
                for servo_id in self.servo_ids:
                    angle = self.teacher_reader.read_angle(servo_id)
                    if angle is not None:
                        teacher_angles[servo_id] = angle
                
                # 控制学生端舵机
                for servo_id, angle in teacher_angles.items():
                    # 检查角度变化，减少不必要的通信
                    if abs(angle - self.last_angles[servo_id]) > 1.0:  # 角度变化超过1度才更新
                        # 将教师端角度映射到学生端范围(-90到90度)
                        mapped_angle = max(-90, min(90, angle))
                        
                        if self.student_controller.set_servo_angle(servo_id, mapped_angle):
                            self.last_angles[servo_id] = angle
                            print(f"\r舵机{servo_id}: {angle:.1f}° -> {mapped_angle:.1f}°", end=" | ")
                        else:
                            print(f"\r舵机{servo_id}: 控制失败", end=" | ")
                
                print()  # 换行
                time.sleep(0.1)  # 控制频率
                
            except Exception as e:
                print(f"\n遥操作过程中发生错误: {e}")
                time.sleep(0.5)
    
    def start_operation(self):
        """开始遥操作"""
        if self.is_running:
            print("遥操作已在运行中")
            return
        
        if not self.teacher_reader or not self.student_controller:
            print("请先连接设备")
            return
        
        self.is_running = True
        self.operation_thread = threading.Thread(target=self._operation_loop)
        self.operation_thread.daemon = True
        self.operation_thread.start()
        
    def stop_operation(self):
        """停止遥操作"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.operation_thread:
            self.operation_thread.join(timeout=2)
            self.operation_thread = None
        
        print("\n遥操作已停止")

def main():
    """主函数"""
    # 配置参数
    TEACHER_PORT = "COM8"  # 教师端串口，根据实际情况修改
    STUDENT_PORT = "COM6"  # 学生端串口，根据实际情况修改
    SERVO_IDS = [1, 2, 3, 4, 5, 6]  # 舵机ID列表
    
    # 创建遥操作控制器
    remote_controller = RemoteOperationController(
        teacher_port=TEACHER_PORT,
        student_port=STUDENT_PORT,
        servo_ids=SERVO_IDS
    )
    
    try:
        # 连接设备
        if not remote_controller.connect():
            print("设备连接失败，程序退出")
            return
        
        # 开始遥操作
        remote_controller.start_operation()
        
        print("\n遥操作正在运行...")
        print("按 Ctrl+C 停止程序")
        print("---")
        
        # 保持程序运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序发生错误: {e}")
    finally:
        # 清理资源
        remote_controller.disconnect()
        print("程序结束")

if __name__ == '__main__':
    main()