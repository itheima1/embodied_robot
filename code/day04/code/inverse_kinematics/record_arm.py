import time
import struct
import enum
import logging
import math
import json
import csv
from datetime import datetime
from copy import deepcopy
import numpy as np
from serial import Serial
from serial.tools import list_ports

# region: GBot/utils.py

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

# endregion

# region: GBot/global_state.py

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

# endregion

# region: GBot/port_handler.py

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

# endregion

# region: GBot/sync_connector.py

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

# endregion

# region: 机械臂动作记录器

class ArmMotionRecorder:
    def __init__(self, port: str, joint_ids: list[int] = [1, 2, 3, 4, 5, 6]):
        self.port = port
        self.joint_ids = joint_ids
        self.__port_handler: PortHandler = PortHandler()
        self.__sync_connector: SyncConnector = SyncConnector(self.__port_handler)
        self.is_connected = False
        # 舵机参数配置
        self.homing_offset = 2048  # 零位偏移
        self.resolution = 4096     # 分辨率
        
        # 记录相关
        self.is_recording = False
        self.recorded_data = []
        self.start_time = None
        
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
            print("请先连接到串口")
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
                    print(f"舵机ID {motor_id} 未返回位置数据")
                    return None
            else:
                print(f"读取舵机ID {motor_id} 失败，错误码: {result.get_error_code()}")
                return None
                
        except Exception as e:
            print(f"读取舵机ID {motor_id} 时发生异常: {e}")
            return None
    
    def read_all_angles(self) -> dict:
        """读取所有关节的角度
        
        Returns:
            dict: 关节ID到角度的映射
        """
        angles = {}
        for joint_id in self.joint_ids:
            angle = self.read_angle(joint_id)
            angles[joint_id] = angle
        return angles
    
    def start_recording(self):
        """开始记录动作序列"""
        if self.is_recording:
            print("已经在记录中")
            return
        
        self.is_recording = True
        self.recorded_data = []
        self.start_time = time.time()
        print("开始记录机械臂动作序列...")
        print("按 Ctrl+C 停止记录")
    
    def stop_recording(self):
        """停止记录动作序列"""
        if not self.is_recording:
            print("当前未在记录")
            return
        
        self.is_recording = False
        print(f"\n记录停止，共记录了 {len(self.recorded_data)} 个数据点")
    
    def record_frame(self):
        """记录一帧数据"""
        if not self.is_recording:
            return
        
        # 读取所有关节角度
        angles = self.read_all_angles()
        
        # 计算时间戳
        timestamp = time.time() - self.start_time
        
        # 创建数据帧
        frame = {
            'timestamp': timestamp,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'angles': angles
        }
        
        self.recorded_data.append(frame)
        
        # 打印当前状态
        angle_str = " | ".join([f"J{joint_id}: {angle:.2f}°" if angle is not None else f"J{joint_id}: ERROR" 
                                for joint_id, angle in angles.items()])
        print(f"[{timestamp:.2f}s] {angle_str}")
    
    def save_to_json(self, filename: str = None):
        """保存记录数据到JSON文件"""
        if not self.recorded_data:
            print("没有记录数据可保存")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"arm_motion_{timestamp}.json"
        
        # 准备保存的数据
        save_data = {
            'metadata': {
                'joint_ids': self.joint_ids,
                'total_frames': len(self.recorded_data),
                'duration': self.recorded_data[-1]['timestamp'] if self.recorded_data else 0,
                'recorded_at': datetime.now().isoformat()
            },
            'motion_data': self.recorded_data
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"数据已保存到: {filename}")
        except Exception as e:
            print(f"保存JSON文件失败: {e}")
    
    def save_to_csv(self, filename: str = None):
        """保存记录数据到CSV文件"""
        if not self.recorded_data:
            print("没有记录数据可保存")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"arm_motion_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                # 创建CSV写入器
                fieldnames = ['timestamp', 'datetime'] + [f'joint_{joint_id}_angle' for joint_id in self.joint_ids]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 写入数据
                for frame in self.recorded_data:
                    row = {
                        'timestamp': frame['timestamp'],
                        'datetime': frame['datetime']
                    }
                    for joint_id in self.joint_ids:
                        row[f'joint_{joint_id}_angle'] = frame['angles'].get(joint_id)
                    writer.writerow(row)
                
            print(f"数据已保存到: {filename}")
        except Exception as e:
            print(f"保存CSV文件失败: {e}")

# endregion

def main():
    """主函数 - 记录机械臂6个关节的动作序列"""
    # 配置参数
    port = 'COM6'  # 根据实际情况修改串口
    joint_ids = [1, 2, 3, 4, 5, 6]  # 机械臂6个关节的ID
    
    # 创建机械臂动作记录器
    recorder = ArmMotionRecorder(port, joint_ids)
    
    try:
        # 连接到串口
        if not recorder.connect():
            return
        
        print("机械臂动作记录器已启动")
        print("命令说明:")
        print("  's' - 开始记录")
        print("  'q' - 停止记录并退出")
        print("  其他 - 显示当前角度")
        print("---")
        
        while True:
            if recorder.is_recording:
                # 记录模式：自动记录数据
                recorder.record_frame()
                time.sleep(0.1)  # 每100ms记录一次
            else:
                # 等待用户输入
                try:
                    cmd = input("请输入命令 (s=开始记录, q=退出): ").strip().lower()
                    
                    if cmd == 's':
                        recorder.start_recording()
                    elif cmd == 'q':
                        break
                    else:
                        # 显示当前角度
                        angles = recorder.read_all_angles()
                        angle_str = " | ".join([f"关节{joint_id}: {angle:.2f}°" if angle is not None else f"关节{joint_id}: 读取失败" 
                                               for joint_id, angle in angles.items()])
                        print(f"当前角度: {angle_str}")
                        
                except EOFError:
                    break
            
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，停止记录...")
        if recorder.is_recording:
            recorder.stop_recording()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 保存数据并断开连接
        if recorder.recorded_data:
            print("\n保存记录数据...")
            recorder.save_to_json()
            recorder.save_to_csv()
        
        recorder.disconnect()
        print("程序结束")

if __name__ == '__main__':
    main()