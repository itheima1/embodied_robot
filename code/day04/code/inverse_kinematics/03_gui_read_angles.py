import time
import struct
import enum
import logging
import math
from copy import deepcopy
import numpy as np
from serial import Serial
from serial.tools import list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import threading

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

# region: 简化的舵机角度读取类

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
        return True

    def disconnect(self):
        """断开串口连接"""
        if not self.is_connected:
            return
        
        self.__port_handler.close()
        self.is_connected = False

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

# endregion

# region: GUI应用程序

class ServoAngleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("机械臂关节角度监控")
        self.root.geometry("800x600")
        
        # 配置参数
        self.port = 'COM6'  # 根据实际情况修改串口
        self.motor_ids = [1, 2, 3, 4, 5, 6]  # 要读取的舵机ID列表
        
        # 数据存储
        self.angles = {motor_id: 0.0 for motor_id in self.motor_ids}
        self.is_reading = False
        self.reader = None
        
        # 创建界面
        self.create_widgets()
        
        # 创建图表
        self.create_chart()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """创建界面控件"""
        # 控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # 串口设置
        ttk.Label(control_frame, text="串口:").grid(row=0, column=0, padx=5)
        self.port_var = tk.StringVar(value=self.port)
        port_entry = ttk.Entry(control_frame, textvariable=self.port_var, width=10)
        port_entry.grid(row=0, column=1, padx=5)
        
        # 控制按钮
        self.start_button = ttk.Button(control_frame, text="开始读取", command=self.start_reading)
        self.start_button.grid(row=0, column=2, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="停止读取", command=self.stop_reading, state='disabled')
        self.stop_button.grid(row=0, column=3, padx=5)
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="状态: 未连接", foreground="red")
        self.status_label.grid(row=0, column=4, padx=20)
    
    def create_chart(self):
        """创建图表"""
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor('white')
        
        # 设置图表
        self.ax.set_title('机械臂关节角度实时监控', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('关节ID', fontsize=12)
        self.ax.set_ylabel('角度 (度)', fontsize=12)
        self.ax.set_ylim(-180, 180)
        self.ax.grid(True, alpha=0.3)
        
        # 创建柱状图
        self.bars = self.ax.bar(self.motor_ids, [0] * len(self.motor_ids), 
                               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
                               alpha=0.8, edgecolor='black', linewidth=1)
        
        # 设置x轴标签
        self.ax.set_xticks(self.motor_ids)
        self.ax.set_xticklabels([f'关节{i}' for i in self.motor_ids])
        
        # 添加数值标签
        self.value_labels = []
        for i, bar in enumerate(self.bars):
            label = self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               '0.0°', ha='center', va='bottom', fontweight='bold')
            self.value_labels.append(label)
        
        # 将图表嵌入到tkinter窗口
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def start_reading(self):
        """开始读取角度"""
        if self.is_reading:
            return
        
        # 更新串口
        self.port = self.port_var.get()
        
        # 创建读取器并连接
        self.reader = ServoAngleReader(self.port)
        if not self.reader.connect():
            messagebox.showerror("错误", f"无法连接到串口 {self.port}")
            return
        
        # 更新状态
        self.is_reading = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="状态: 正在读取", foreground="green")
        
        # 启动读取线程
        self.reading_thread = threading.Thread(target=self.reading_loop, daemon=True)
        self.reading_thread.start()
        
        # 启动图表更新
        self.update_chart()
    
    def stop_reading(self):
        """停止读取角度"""
        if not self.is_reading:
            return
        
        # 更新状态
        self.is_reading = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="状态: 已停止", foreground="orange")
        
        # 断开连接
        if self.reader:
            self.reader.disconnect()
            self.reader = None
    
    def reading_loop(self):
        """读取循环（在后台线程中运行）"""
        while self.is_reading:
            try:
                # 读取所有舵机的角度
                for motor_id in self.motor_ids:
                    if not self.is_reading:
                        break
                    angle = self.reader.read_angle(motor_id)
                    if angle is not None:
                        self.angles[motor_id] = angle
                
                time.sleep(0.1)  # 每0.1秒读取一次
                
            except Exception as e:
                print(f"读取错误: {e}")
                break
    
    def update_chart(self):
        """更新图表"""
        if not self.is_reading:
            return
        
        try:
            # 更新柱状图高度和标签
            for i, (motor_id, bar) in enumerate(zip(self.motor_ids, self.bars)):
                angle = self.angles[motor_id]
                bar.set_height(angle)
                
                # 更新数值标签位置和文本
                label = self.value_labels[i]
                label.set_position((bar.get_x() + bar.get_width()/2, 
                                  angle + (10 if angle >= 0 else -15)))
                label.set_text(f'{angle:.1f}°')
                
                # 根据角度设置颜色
                if abs(angle) > 150:
                    bar.set_color('#FF4444')  # 红色 - 危险角度
                elif abs(angle) > 90:
                    bar.set_color('#FFA500')  # 橙色 - 警告角度
                else:
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                    bar.set_color(colors[i % len(colors)])  # 正常颜色
            
            # 重新绘制图表
            self.canvas.draw()
            
        except Exception as e:
            print(f"图表更新错误: {e}")
        
        # 继续更新（每100ms更新一次图表）
        if self.is_reading:
            self.root.after(100, self.update_chart)
    
    def on_closing(self):
        """窗口关闭事件"""
        self.stop_reading()
        self.root.destroy()

# endregion

def main():
    """主函数"""
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建主窗口
    root = tk.Tk()
    app = ServoAngleGUI(root)
    
    # 运行应用程序
    root.mainloop()

if __name__ == '__main__':
    main()