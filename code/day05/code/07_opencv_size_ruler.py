import cv2
import numpy as np
import json
import math
from collections import deque

class SizeRuler:
    def __init__(self):
        # 黑色的HSV颜色范围
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 30])
        
        # 均值滤波参数
        self.filter_window_size = 5  # 滤波窗口大小
        self.position_history = deque(maxlen=self.filter_window_size)  # 位置历史记录
        self.filtered_position = None  # 滤波后的位置
        
        # 标定参数
        self.reference_size_cm = 3.0  # 参考正方形的真实尺寸（厘米）
        self.pixels_per_cm = None  # 像素/厘米比例
        self.is_calibrated = False  # 是否已标定
        
        # 鼠标测量相关
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.measurement_lines = []  # 存储测量线段
        
    def apply_position_filter(self, current_position):
        """应用均值滤波到位置数据"""
        if current_position is None:
            return self.filtered_position
        
        # 添加当前位置到历史记录
        self.position_history.append(current_position)
        
        # 计算均值滤波后的位置
        if len(self.position_history) > 0:
            positions_array = np.array(self.position_history)
            self.filtered_position = np.mean(positions_array, axis=0).astype(int)
        
        return self.filtered_position
    
    def calculate_calibration(self, contour):
        """基于检测到的黑色正方形计算标定参数"""
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 使用边界框的平均边长作为像素尺寸
        pixel_size = (w + h) / 2
        
        # 计算像素/厘米比例
        self.pixels_per_cm = pixel_size / self.reference_size_cm
        self.is_calibrated = True
        
        return pixel_size
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                # 保存测量线段
                if self.start_point and self.end_point:
                    line_length_pixels = math.sqrt(
                        (self.end_point[0] - self.start_point[0])**2 + 
                        (self.end_point[1] - self.start_point[1])**2
                    )
                    if self.is_calibrated:
                        line_length_cm = line_length_pixels / self.pixels_per_cm
                        self.measurement_lines.append({
                            'start': self.start_point,
                            'end': self.end_point,
                            'length_pixels': line_length_pixels,
                            'length_cm': line_length_cm
                        })
                        print(f"测量结果: {line_length_cm:.2f} cm ({line_length_pixels:.1f} 像素)")
                    else:
                        print("请先检测到黑色正方形进行标定")
    
    def detect_black_blocks(self, frame):
        """检测帧中的黑色木块并返回位置信息"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建黑色掩码
        mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        current_position = None
        
        # 找到最大的轮廓作为主要目标
        max_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500 and area > max_area:
                max_area = area
                max_contour = contour
        
        if max_contour is not None:
            # 计算轮廓的中心点
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_position = [cx, cy]
                
                # 应用均值滤波
                filtered_pos = self.apply_position_filter(current_position)
                
                if filtered_pos is not None:
                    fx, fy = filtered_pos
                    
                    # 添加到结果中（使用滤波后的位置）
                    results.append({
                        "type": "color",
                        "value": "black",
                        "position_pixels": [int(fx), int(fy)]
                    })
                    
                    # 在图像上绘制检测结果
                    # 绘制原始检测点（红色）
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f'raw({cx},{cy})', 
                              (cx-40, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.4, (0, 0, 255), 1)
                    
                    # 绘制滤波后的点（绿色）
                    cv2.circle(frame, (fx, fy), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f'filtered({fx},{fy})', 
                              (fx-60, fy-20), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 255), 2)
                    
                    # 计算标定参数
                    pixel_size = self.calculate_calibration(max_contour)
                    
                    # 绘制轮廓
                    cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                    
                    # 显示标定信息
                    cv2.putText(frame, f'Calibrated: {pixel_size:.1f}px = {self.reference_size_cm}cm', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f'Scale: {self.pixels_per_cm:.2f} px/cm', 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # 如果没有检测到目标，清空滤波器
            self.apply_position_filter(None)
        
        # 绘制测量线段
        for line in self.measurement_lines:
            cv2.line(frame, line['start'], line['end'], (255, 0, 255), 2)
            # 在线段中点显示长度
            mid_x = (line['start'][0] + line['end'][0]) // 2
            mid_y = (line['start'][1] + line['end'][1]) // 2
            cv2.putText(frame, f'{line["length_cm"]:.2f}cm', 
                      (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 绘制当前正在画的线
        if self.drawing and self.start_point and self.end_point:
            cv2.line(frame, self.start_point, self.end_point, (0, 255, 255), 2)
            if self.is_calibrated:
                current_length_pixels = math.sqrt(
                    (self.end_point[0] - self.start_point[0])**2 + 
                    (self.end_point[1] - self.start_point[1])**2
                )
                current_length_cm = current_length_pixels / self.pixels_per_cm
                cv2.putText(frame, f'{current_length_cm:.2f}cm', 
                          self.end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return results, frame
    
    def run(self):
        """运行尺寸测量工具"""
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        print("智能尺寸测量工具已启动")
        print("使用说明：")
        print("1. 将3cm×3cm的黑色正方形放入视野进行标定")
        print("2. 标定完成后，用鼠标拖拽画线测量物体尺寸")
        print("3. 按 'c' 键清除所有测量线段")
        print("4. 按 'q' 键退出程序")
        print(f"使用HSV参数 - Lower: {self.black_lower}, Upper: {self.black_upper}")
        print("-" * 50)
        
        # 设置鼠标回调
        cv2.namedWindow('ruler_size')
        cv2.setMouseCallback('ruler_size', self.mouse_callback)
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头数据")
                break
            
            # 检测黑色木块
            results, processed_frame = self.detect_black_blocks(frame)
            
            # 输出检测结果
            if results:
                print(f"检测结果: {json.dumps(results, ensure_ascii=False)}")
            
            # 显示使用说明
            if not self.is_calibrated:
                cv2.putText(processed_frame, 'Place 3cm x 3cm black square for calibration', 
                          (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0, 0, 255), 2)
            else:
                cv2.putText(processed_frame, 'Drag mouse to measure objects', 
                          (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0, 255, 0), 2)
            
            # 显示处理后的图像
            cv2.imshow('ruler_size', processed_frame)
            
            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # 清除所有测量线段
                self.measurement_lines.clear()
                print("已清除所有测量线段")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

def main():
    """主函数"""
    ruler = SizeRuler()
    ruler.run()

if __name__ == "__main__":
    main()