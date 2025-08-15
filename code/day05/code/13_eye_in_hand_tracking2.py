#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机械臂视觉追踪系统
结合红色积木块检测和机械臂控制，实现自动追踪功能

功能说明：
- 实时检测红色积木块位置
- 计算目标与屏幕中心的偏差
- 自动控制机械臂移动保持目标在屏幕中央
- 支持手动控制模式切换

控制说明：
- 空格键: 切换自动/手动模式
- 手动模式下：
  - W/S: 前进/后退 (X轴)
  - A/D: 左移/右移 (Y轴) 
  - Q/E: 上升/下降 (Z轴)
  - R: 重置到初始位置
- C: 清除轨迹
- ESC: 退出程序
"""

import cv2
import numpy as np
import json
import time
import sys
import os
from collections import deque

# 添加robot目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'robot'))

try:
    from robot_arm_control import RobotArmController
except ImportError:
    print("错误：无法导入机械臂控制模块")
    print("请确保robot_arm_control.py文件存在于robot目录中")
    sys.exit(1)

class RedCapTracker:
    """红色积木块检测器"""
    
    def __init__(self):
        # 红色HSV颜色范围
        self.red_lower = np.array([0, 133, 163])
        self.red_upper = np.array([29, 163, 198])
        
        # 轨迹追踪参数
        self.track_points = deque(maxlen=50)
        
        # 形态学操作核
        self.kernel = np.ones((5, 5), np.uint8)
        
        # 最小检测面积
        self.min_area = 300
        
    def get_red_mask(self, hsv_image):
        """获取红色掩码"""
        mask = cv2.inRange(hsv_image, self.red_lower, self.red_upper)
        return mask
    
    def morphological_operations(self, mask):
        """形态学操作：去除噪声并填补空洞"""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        return mask
    
    def is_valid_target(self, contour):
        """判断轮廓是否为有效目标"""
        area = cv2.contourArea(contour)
        return area >= self.min_area
    
    def detect_red_cap(self, frame):
        """检测红色积木块"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # 获取红色掩码
        red_mask = self.get_red_mask(hsv)
        processed_mask = self.morphological_operations(red_mask)
        
        # 查找轮廓
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cap = None
        best_area = 0
        
        # 寻找最佳的红色目标
        for contour in contours:
            if self.is_valid_target(contour):
                area = cv2.contourArea(contour)
                if area > best_area:
                    best_area = area
                    best_cap = contour
        
        result = None
        if best_cap is not None:
            # 计算中心点
            M = cv2.moments(best_cap)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 添加到轨迹
                self.track_points.append((cx, cy))
                
                # 创建结果
                result = {
                    "type": "track_target",
                    "position_pixels": [cx, cy],
                    "area": best_area
                }
                
                # 在图像上绘制检测结果
                self.draw_detection(frame, best_cap, cx, cy)
        
        # 绘制轨迹
        self.draw_trajectory(frame)
        
        return result, frame
    
    def draw_detection(self, frame, contour, cx, cy):
        """在图像上绘制检测结果"""
        # 绘制轮廓
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # 绘制中心点
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
        
        # 绘制外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (255, 0, 0), 2)
        
        # 显示坐标信息
        cv2.putText(frame, f'Target: ({cx}, {cy})', 
                   (cx - 80, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
    
    def draw_trajectory(self, frame):
        """绘制移动轨迹"""
        if len(self.track_points) < 2:
            return
            
        # 绘制轨迹线
        for i in range(1, len(self.track_points)):
            alpha = i / len(self.track_points)
            color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
            cv2.line(frame, self.track_points[i-1], self.track_points[i], color, 2)
            cv2.circle(frame, self.track_points[i], 3, color, -1)
    
    def clear_trajectory(self):
        """清除轨迹"""
        self.track_points.clear()

class EyeInHandTracker:
    """机械臂视觉追踪控制器"""
    
    def __init__(self, port="COM6"):
        # 初始化红色积木块检测器
        self.tracker = RedCapTracker()
        
        # 初始化机械臂控制器
        try:
            self.robot_controller = RobotArmController(port=port)
        except Exception as e:
            print(f"机械臂初始化失败: {e}")
            self.robot_controller = None
        
        # 摄像头参数
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # 控制参数
        self.auto_mode = False  # 自动追踪模式
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.dead_zone = 30  # 死区范围，避免频繁微调
        
        # 移动参数
        self.last_move_time = 0
        self.move_interval = 0.5  # 移动间隔时间（秒）
        
        # 状态显示
        self.status_text = "手动模式"
        
    def init_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(2)
        
        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("摄像头初始化成功")
        return True
    
    def process_frame(self, frame):
        """处理摄像头帧"""
        # 画面方向修正
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rotated_frame = cv2.flip(rotated_frame, 1)
        rotated_frame = cv2.flip(rotated_frame, 0)
        
        # 检测红色积木块
        result, processed_frame = self.tracker.detect_red_cap(rotated_frame)
        
        # 绘制屏幕中心十字线
        self.draw_center_cross(processed_frame)
        
        # 绘制状态信息
        self.draw_status_info(processed_frame, result)
        
        return result, processed_frame
    
    def draw_center_cross(self, frame):
        """绘制屏幕中心十字线"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 绘制十字线
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        
        # 绘制死区范围
        cv2.rectangle(frame, 
                     (center_x - self.dead_zone, center_y - self.dead_zone),
                     (center_x + self.dead_zone, center_y + self.dead_zone),
                     (255, 255, 0), 1)
    
    def draw_status_info(self, frame, detection_result):
        """绘制状态信息"""
        h, w = frame.shape[:2]
        
        # 模式状态
        mode_text = "自动追踪" if self.auto_mode else "手动控制"
        mode_color = (0, 255, 0) if self.auto_mode else (0, 0, 255)
        cv2.putText(frame, f'Mode: {mode_text}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # 检测状态
        if detection_result:
            target_x, target_y = detection_result["position_pixels"]
            cv2.putText(frame, f'Target: ({target_x}, {target_y})', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 计算偏差
            center_x, center_y = w // 2, h // 2
            offset_x = target_x - center_x
            offset_y = target_y - center_y
            cv2.putText(frame, f'Offset: ({offset_x}, {offset_y})', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, 'Target: Not Found', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 控制说明
        controls = [
            "SPACE: Toggle Auto/Manual",
            "Manual: WASD QE R",
            "C: Clear trajectory",
            "ESC: Exit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (10, h - 100 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def auto_track_control(self, detection_result, frame_shape):
        """自动追踪控制逻辑"""
        if not self.auto_mode or not detection_result or not self.robot_controller:
            return
        
        # 检查移动间隔
        current_time = time.time()
        if current_time - self.last_move_time < self.move_interval:
            return
        
        target_x, target_y = detection_result["position_pixels"]
        
        # 获取当前帧尺寸（处理后的帧）
        h, w = frame_shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 计算偏差
        offset_x = target_x - center_x
        offset_y = target_y - center_y
        
        # 检查是否在死区内
        if abs(offset_x) < self.dead_zone and abs(offset_y) < self.dead_zone:
            return
        
        # 控制机械臂移动
        moved = False
        
        # X轴控制（左右移动）
        if offset_x > self.dead_zone:  # 目标在右边，机械臂向右移动
            success = self.robot_controller.execute_movement("右移", 
                     self.robot_controller.solver.move_right)
            if success:
                moved = True
                print(f"\r自动追踪: 目标偏右({offset_x}), 机械臂右移", end="")
        elif offset_x < -self.dead_zone:  # 目标在左边，机械臂向左移动
            success = self.robot_controller.execute_movement("左移", 
                     self.robot_controller.solver.move_left)
            if success:
                moved = True
                print(f"\r自动追踪: 目标偏左({offset_x}), 机械臂左移", end="")
        
        # Y轴控制（上下移动）
        if offset_y > self.dead_zone:  # 目标在下方，机械臂向下移动
            success = self.robot_controller.execute_movement("下降", 
                     self.robot_controller.solver.move_down)
            if success:
                moved = True
                print(f"\r自动追踪: 目标偏下({offset_y}), 机械臂下移", end="")
        elif offset_y < -self.dead_zone:  # 目标在上方，机械臂向上移动
            success = self.robot_controller.execute_movement("上升", 
                     self.robot_controller.solver.move_up)
            if success:
                moved = True
                print(f"\r自动追踪: 目标偏上({offset_y}), 机械臂上移", end="")
        
        if moved:
            self.last_move_time = current_time
    
    def manual_control(self, key):
        """手动控制逻辑"""
        if self.auto_mode or not self.robot_controller:
            return
        
        if key == ord('w') or key == ord('W'):
            self.robot_controller.execute_movement("前进", 
                   self.robot_controller.solver.move_forward)
        elif key == ord('s') or key == ord('S'):
            self.robot_controller.execute_movement("后退", 
                   self.robot_controller.solver.move_backward)
        elif key == ord('a') or key == ord('A'):
            self.robot_controller.execute_movement("左移", 
                   self.robot_controller.solver.move_left)
        elif key == ord('d') or key == ord('D'):
            self.robot_controller.execute_movement("右移", 
                   self.robot_controller.solver.move_right)
        elif key == ord('q') or key == ord('Q'):
            self.robot_controller.execute_movement("上升", 
                   self.robot_controller.solver.move_up)
        elif key == ord('e') or key == ord('E'):
            self.robot_controller.execute_movement("下降", 
                   self.robot_controller.solver.move_down)
        elif key == ord('r') or key == ord('R'):
            print("\n重置到初始位置...")
            self.robot_controller.current_angles = np.deg2rad([0, 30, -30, -30, 0])
            self.robot_controller.move_to_initial_position()
    
    def run(self):
        """主运行循环"""
        print("=== 机械臂视觉追踪系统启动 ===")
        print("初始化摄像头...")
        
        if not self.init_camera():
            return
        
        print("系统启动完成")
        print("按空格键切换自动/手动模式")
        print("手动模式: WASD QE控制移动，R重置")
        print("自动模式: 系统自动追踪红色积木块")
        print("按ESC退出程序")
        print("-" * 50)
        
        try:
            while True:
                # 读取摄像头帧
                ret, frame = self.cap.read()
                if not ret:
                    print("错误：无法读取摄像头画面")
                    break
                
                # 处理帧
                detection_result, processed_frame = self.process_frame(frame)
                
                # 自动追踪控制
                self.auto_track_control(detection_result, processed_frame.shape)
                
                # 显示处理后的画面
                cv2.imshow('Eye-in-Hand Tracking System', processed_frame)
                
                # 检测按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC键
                    break
                elif key == ord(' '):  # 空格键切换模式
                    self.auto_mode = not self.auto_mode
                    mode_text = "自动追踪" if self.auto_mode else "手动控制"
                    print(f"\n切换到{mode_text}模式")
                elif key == ord('c') or key == ord('C'):
                    self.tracker.clear_trajectory()
                    print("\n轨迹已清除")
                else:
                    # 手动控制
                    self.manual_control(key)
                
                # 检查窗口是否被关闭
                if cv2.getWindowProperty('Eye-in-Hand Tracking System', cv2.WND_PROP_VISIBLE) < 1:
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
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.robot_controller:
            self.robot_controller.cleanup()
        
        print("程序结束")

def main():
    """主函数"""
    # 配置参数
    ROBOT_PORT = "COM6"  # 根据实际情况调整串口
    
    try:
        # 创建视觉追踪控制器
        tracker = EyeInHandTracker(port=ROBOT_PORT)
        
        # 运行追踪程序
        tracker.run()
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        print("请检查：")
        print("1. 摄像头连接是否正确")
        print("2. 机械臂串口连接是否正确")
        print("3. 相关依赖文件是否存在")

if __name__ == '__main__':
    main()