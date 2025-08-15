from mcp.server.fastmcp import FastMCP
import serial
import time
import sys

# 创建MCP服务器实例
mcp = FastMCP()

# --- ServoController Class ---
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

# 全局机械臂控制器实例
robot_controller = None

def initialize_robot():
    """初始化机械臂控制器"""
    global robot_controller
    if robot_controller is None:
        try:
            robot_controller = ServoController(port="COM6", baudrate=1000000)
            # 启用所有舵机的扭矩
            for servo_id in range(1, 7):
                robot_controller.enable_torque(servo_id)
                time.sleep(0.05)
            print("机械臂初始化成功")
            return True
        except Exception as e:
            print(f"机械臂初始化失败: {e}")
            return False
    return True

#### MCP工具函数 ####

# 添加加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加"""
    print(f"计算 {a} 加 {b}")
    return a + b

# 腰部关节旋转控制工具（1号舵机）
@mcp.tool()
def rotate_waist(angle: float) -> str:
    """控制机械臂腰部关节旋转
    
    Args:
        angle: 旋转角度，范围-90度到90度
    
    Returns:
        str: 执行结果信息
    """
    # 检查角度范围
    if angle < -90 or angle > 90:
        return f"错误：角度 {angle} 超出范围，腰部关节旋转范围为-90度到90度"
    
    # 初始化机械臂
    if not initialize_robot():
        return "错误：机械臂初始化失败"
    
    try:
        # 控制1号舵机（腰部）
        success = robot_controller.set_servo_angle(1, angle)
        if success:
            print(f"控制腰部关节旋转到 {angle} 度")
            return f"腰部关节已旋转到 {angle} 度"
        else:
            return f"错误：腰部关节旋转到 {angle} 度失败"
    except Exception as e:
        return f"错误：腰部关节控制异常 - {str(e)}"

# 夹爪控制工具（6号舵机）
@mcp.tool()
def control_gripper(angle: float) -> str:
    """控制机械臂夹爪开合
    
    Args:
        angle: 夹爪角度，范围0度到-90度（0度为完全张开，-90度为完全闭合）
    
    Returns:
        str: 执行结果信息
    """
    # 检查角度范围
    if angle > 0 or angle < -90:
        return f"错误：角度 {angle} 超出范围，夹爪控制范围为0度到-90度"
    
    # 初始化机械臂
    if not initialize_robot():
        return "错误：机械臂初始化失败"
    
    try:
        # 控制6号舵机（夹爪）
        success = robot_controller.set_servo_angle(6, angle)
        if success:
            print(f"控制夹爪到 {angle} 度位置")
            if angle == 0:
                status = "完全张开"
            elif angle == -90:
                status = "完全闭合"
            else:
                status = f"部分闭合（{abs(angle)}度）"
            
            return f"夹爪已调整到 {angle} 度位置（{status}）"
        else:
            return f"错误：夹爪控制到 {angle} 度失败"
    except Exception as e:
        return f"错误：夹爪控制异常 - {str(e)}"

# 控制指定舵机角度
@mcp.tool()
def control_servo(servo_id: int, angle: float) -> str:
    """控制指定舵机的角度
    
    Args:
        servo_id: 舵机ID，范围1-6
        angle: 角度，范围-90度到90度
    
    Returns:
        str: 执行结果信息
    """
    # 检查舵机ID范围
    if servo_id < 1 or servo_id > 6:
        return f"错误：舵机ID {servo_id} 超出范围，舵机ID范围为1-6"
    
    # 检查角度范围
    if angle < -90 or angle > 90:
        return f"错误：角度 {angle} 超出范围，角度范围为-90度到90度"
    
    # 初始化机械臂
    if not initialize_robot():
        return "错误：机械臂初始化失败"
    
    try:
        success = robot_controller.set_servo_angle(servo_id, angle)
        if success:
            print(f"控制{servo_id}号舵机到 {angle} 度")
            return f"{servo_id}号舵机已调整到 {angle} 度"
        else:
            return f"错误：{servo_id}号舵机控制到 {angle} 度失败"
    except Exception as e:
        return f"错误：{servo_id}号舵机控制异常 - {str(e)}"

# 机械臂复位
@mcp.tool()
def reset_robot() -> str:
    """将机械臂所有关节复位到0度位置
    
    Returns:
        str: 执行结果信息
    """
    # 初始化机械臂
    if not initialize_robot():
        return "错误：机械臂初始化失败"
    
    try:
        success_count = 0
        for servo_id in range(1, 7):
            success = robot_controller.set_servo_angle(servo_id, 0)
            if success:
                success_count += 1
            time.sleep(0.1)  # 添加延时避免指令冲突
        
        print(f"机械臂复位完成，成功控制 {success_count}/6 个舵机")
        if success_count == 6:
            return "机械臂已成功复位到初始位置（所有关节0度）"
        else:
            return f"机械臂部分复位完成，成功控制 {success_count}/6 个舵机"
    except Exception as e:
        return f"错误：机械臂复位异常 - {str(e)}"

if __name__ == "__main__":
    try:
        # 初始化并运行服务器
        print("启动机械臂MCP服务器...")
        mcp.run(transport='sse')
    except KeyboardInterrupt:
        print("\n服务器停止")
    finally:
        # 清理资源
        if robot_controller:
            robot_controller.close()
        print("程序结束")