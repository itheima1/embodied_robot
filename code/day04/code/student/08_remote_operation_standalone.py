#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立版遥操作控制程序
实现学生端舵机实时复制教师端舵机的动作
包含所有必要的类，无需依赖其他文件
"""

import time
import struct
import enum
import sys
import serial
from serial import Serial

# ===== 教师端读取相关代码 =====

def bytes_to_short(data: bytes, signed: bool = False, byteorder: str = 'little') -> int:
    if len(data) != 2:
        raise ValueError("Data must be exactly 2 bytes long")
    prefix = '<' if byteorder == 'little' else '>'
    format_char = 'h' if signed else 'H'
    return struct.unpack(f"{prefix}{format_char}", data)[0]

class Address(enum.Enum):
    CURRENT_POSITION    = (7, 2)
    TORQUE_ENABLE       = (50, 1)
    TARGET_POSITION     = (51, 2)

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
        self._read_timeout = 1
        self._write_timeout = None
        self.__is_running = False

    def open(self, port) -> bool:
        self.close()
        try:
            self._port = port
            self.__serial = Serial(
                port=port, 
                baudrate=self._baudrate, 
                bytesize=self._bytesize, 
                parity=self._parity, 
                stopbits=self._stopbits, 
                timeout=self._read_timeout, 
                write_timeout=self._write_timeout
            )
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

FRAME_HEADER = 0xAA
FRAME_TAIL = 0xBB
FRAME_CMD_READ = 0x03

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
        self.last_read_address = None

    def _parse_response_frame(self) -> Result:
        retry_cnt = 0
        read_list = []
        state = 0
        
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

class TeacherServoReader:
    """教师端舵机角度读取器"""
    def __init__(self, port: str):
        self.port = port
        self.__port_handler = PortHandler()
        self.__sync_connector = SyncConnector(self.__port_handler)
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
        return True

    def disconnect(self):
        """断开串口连接"""
        if not self.is_connected:
            return
        
        self.__port_handler.close()
        self.is_connected = False

    def read_angle(self, motor_id: int) -> float:
        """读取指定ID舵机的角度"""
        if not self.is_connected:
            return None
        
        try:
            result = self.__sync_connector.read([motor_id], [Address.CURRENT_POSITION])
            
            if result.is_success():
                raw_position = result.get_data(Address.CURRENT_POSITION)
                if raw_position is not None:
                    # 转换为角度
                    angle = ((raw_position - self.homing_offset) / self.resolution) * 360
                    return angle
            return None
                
        except Exception:
            return None

# ===== 学生端控制相关代码 =====

class StudentServoController:
    """学生端舵机控制器"""
    def __init__(self, port="COM6", baudrate=1000000, timeout=0.1):
        self.port_name = port
        self.serial_port = None
        # Constants
        self.ADDR_GOAL_POSITION = 42
        self.ADDR_TORQUE_ENABLE = 40
        self.INST_WRITE = 3

        try:
            self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            time.sleep(0.1)
            self.serial_port.reset_input_buffer()
        except serial.SerialException as e:
            raise e

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

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
        
        return self._send_packet(servo_id, self.INST_WRITE, params)

    def enable_torque(self, servo_id):
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, 1, size=1)

    def set_servo_angle(self, servo_id, angle):
        """设置舵机角度 (-90 到 90 度)"""
        # 将角度映射到位置值 (1024 到 3072)
        position = int(((angle + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        # 限制范围
        position = max(1024, min(3072, position))
        
        return self._write_register(servo_id, self.ADDR_GOAL_POSITION, position, size=2)

# ===== 遥操作主控制器 =====

class RemoteOperationController:
    """遥操作控制器"""
    def __init__(self, teacher_port="COM5", student_port="COM6", servo_ids=None):
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
        
        # 角度缓存和统计
        self.last_angles = {servo_id: 0.0 for servo_id in self.servo_ids}
        self.success_count = 0
        self.error_count = 0
        
    def connect(self):
        """连接教师端和学生端"""
        try:
            # 连接教师端
            print(f"正在连接教师端 ({self.teacher_port})...")
            self.teacher_reader = TeacherServoReader(self.teacher_port)
            if not self.teacher_reader.connect():
                print("❌ 教师端连接失败")
                return False
            print("✅ 教师端连接成功")
            
            # 连接学生端
            print(f"正在连接学生端 ({self.student_port})...")
            self.student_controller = StudentServoController(port=self.student_port, baudrate=1000000)
            print("✅ 学生端连接成功")
            
            # 启用学生端舵机扭矩
            print("正在启用学生端舵机扭矩...")
            for servo_id in self.servo_ids:
                if self.student_controller.enable_torque(servo_id):
                    print(f"  ✅ 舵机{servo_id}扭矩启用成功")
                else:
                    print(f"  ❌ 舵机{servo_id}扭矩启用失败")
                time.sleep(0.1)
            
            print("\n🎉 所有设备连接成功！")
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
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
            
        print("\n🔌 已断开所有连接")
    
    def read_teacher_angles(self):
        """读取教师端所有舵机角度"""
        angles = {}
        for servo_id in self.servo_ids:
            try:
                angle = self.teacher_reader.read_angle(servo_id)
                if angle is not None:
                    angles[servo_id] = angle
            except Exception:
                pass
        return angles
    
    def control_student_servos(self, teacher_angles):
        """控制学生端舵机"""
        results = {}
        for servo_id, angle in teacher_angles.items():
            try:
                # 检查角度变化，减少不必要的通信
                angle_diff = abs(angle - self.last_angles[servo_id])
                if angle_diff > 3.0:  # 角度变化超过3度才更新
                    # 将教师端角度映射到学生端范围(-90到90度)
                    mapped_angle = max(-90, min(90, angle))
                    
                    if self.student_controller.set_servo_angle(servo_id, mapped_angle):
                        self.last_angles[servo_id] = angle
                        results[servo_id] = {
                            'success': True, 
                            'teacher_angle': angle,
                            'student_angle': mapped_angle,
                            'updated': True
                        }
                        self.success_count += 1
                    else:
                        results[servo_id] = {
                            'success': False, 
                            'teacher_angle': angle,
                            'student_angle': mapped_angle,
                            'updated': True
                        }
                        self.error_count += 1
                else:
                    # 角度变化不大，跳过更新
                    results[servo_id] = {
                        'success': True, 
                        'teacher_angle': angle,
                        'student_angle': self.last_angles[servo_id], 
                        'updated': False
                    }
            except Exception as e:
                results[servo_id] = {'success': False, 'error': str(e)}
                self.error_count += 1
        
        return results
    
    def run_operation(self):
        """运行遥操作（阻塞式）"""
        if not self.teacher_reader or not self.student_controller:
            print("❌ 请先连接设备")
            return
        
        self.is_running = True
        print("\n🚀 开始遥操作")
        print("📡 学生端将实时复制教师端动作...")
        print("⏹️  按 Ctrl+C 停止\n")
        
        start_time = time.time()
        cycle_count = 0
        
        try:
            while self.is_running:
                cycle_start = time.time()
                cycle_count += 1
                
                # 读取教师端角度
                teacher_angles = self.read_teacher_angles()
                
                if teacher_angles:
                    # 控制学生端
                    results = self.control_student_servos(teacher_angles)
                    
                    # 显示状态
                    status_parts = []
                    for servo_id in self.servo_ids:
                        if servo_id in results:
                            result = results[servo_id]
                            if result['success']:
                                if result.get('updated', False):
                                    status = f"S{servo_id}:{result['teacher_angle']:.1f}°→{result['student_angle']:.1f}°"
                                else:
                                    status = f"S{servo_id}:保持{result['student_angle']:.1f}°"
                            else:
                                status = f"S{servo_id}:❌"
                        else:
                            status = f"S{servo_id}:无数据"
                        status_parts.append(status)
                    
                    # 计算运行时间和频率
                    elapsed = time.time() - start_time
                    freq = cycle_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\r[{elapsed:.1f}s {freq:.1f}Hz] {' | '.join(status_parts)}", end="")
                else:
                    print("\r❌ 无法读取教师端数据", end="")
                
                # 控制循环频率
                cycle_time = time.time() - cycle_start
                target_cycle_time = 0.1  # 10Hz
                if cycle_time < target_cycle_time:
                    time.sleep(target_cycle_time - cycle_time)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  用户中断操作")
        except Exception as e:
            print(f"\n\n❌ 操作过程中发生错误: {e}")
        finally:
            self.is_running = False
            elapsed = time.time() - start_time
            avg_freq = cycle_count / elapsed if elapsed > 0 else 0
            
            print(f"\n\n📊 操作统计:")
            print(f"  运行时间: {elapsed:.1f}秒")
            print(f"  循环次数: {cycle_count}")
            print(f"  平均频率: {avg_freq:.1f}Hz")
            print(f"  成功次数: {self.success_count}")
            print(f"  失败次数: {self.error_count}")
            if self.success_count + self.error_count > 0:
                success_rate = self.success_count / (self.success_count + self.error_count) * 100
                print(f"  成功率: {success_rate:.1f}%")
    
    def stop_operation(self):
        """停止遥操作"""
        self.is_running = False

def main():
    """主函数"""
    print("🤖 舵机遥操作控制程序")
    print("📋 功能：学生端舵机实时复制教师端舵机动作")
    print("=" * 50)
    
    # 配置参数 - 请根据实际情况修改
    TEACHER_PORT = "COM5"  # 教师端串口
    STUDENT_PORT = "COM6"  # 学生端串口
    SERVO_IDS = [1, 2, 3, 4, 5, 6]  # 舵机ID列表
    
    print(f"⚙️  配置信息:")
    print(f"   教师端串口: {TEACHER_PORT}")
    print(f"   学生端串口: {STUDENT_PORT}")
    print(f"   舵机ID: {SERVO_IDS}")
    print()
    
    # 创建遥操作控制器
    controller = RemoteOperationController(
        teacher_port=TEACHER_PORT,
        student_port=STUDENT_PORT,
        servo_ids=SERVO_IDS
    )
    
    try:
        # 连接设备
        if not controller.connect():
            print("❌ 设备连接失败，程序退出")
            return
        
        # 运行遥操作
        controller.run_operation()
        
    except Exception as e:
        print(f"\n❌ 程序发生错误: {e}")
    finally:
        # 清理资源
        controller.disconnect()
        print("\n👋 程序结束")

if __name__ == '__main__':
    main()