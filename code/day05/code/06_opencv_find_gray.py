import cv2
import numpy as np
import json
from collections import deque

class GrayBlockDetector:
    def __init__(self):
        # 使用优化后的灰色HSV颜色范围
        self.gray_lower = np.array([0, 17, 21])
        self.gray_upper = np.array([84, 39, 79])
        
        # 均值滤波参数
        self.filter_window_size = 5  # 滤波窗口大小
        self.position_history = deque(maxlen=self.filter_window_size)  # 位置历史记录
        self.filtered_position = None  # 滤波后的位置
        
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
    
    def detect_gray_blocks(self, frame):
        """检测帧中的灰色木块并返回位置信息"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建灰色掩码
        mask = cv2.inRange(hsv, self.gray_lower, self.gray_upper)
        
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
                        "value": "gray",
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
                    
                    # 绘制轮廓
                    cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        else:
            # 如果没有检测到目标，清空滤波器
            self.apply_position_filter(None)
        
        return results, frame
    
    def run(self):
        """运行灰色木块检测器"""
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        print("灰色木块检测器已启动")
        print("按 'q' 键退出程序")
        print(f"使用HSV参数 - Lower: {self.gray_lower}, Upper: {self.gray_upper}")
        print("-" * 50)
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头数据")
                break
            
            # 检测灰色木块
            results, processed_frame = self.detect_gray_blocks(frame)
            
            # 输出检测结果
            if results:
                print(f"检测结果: {json.dumps(results, ensure_ascii=False)}")
            
            # 显示处理后的图像
            cv2.imshow('灰色木块检测器', processed_frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

def main():
    """主函数"""
    detector = GrayBlockDetector()
    detector.run()

if __name__ == "__main__":
    main()