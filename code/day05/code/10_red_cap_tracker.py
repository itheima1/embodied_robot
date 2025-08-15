import cv2
import numpy as np
import json
from collections import deque

class RedCapTracker:
    def __init__(self):
        # 红色HSV颜色范围（针对红色瓶盖优化）
        # self.red_lower1 = np.array([0, 50, 50])
        # self.red_upper1 = np.array([10, 255, 255])
        # self.red_lower2 = np.array([170, 50, 50])
        # self.red_upper2 = np.array([180, 255, 255])
        # 红色范围1 (0-10度)
        self.red_lower1 = np.array([1, 211, 93])
        self.red_upper1 = np.array([10, 255, 161])
        # 红色范围2 (170-180度)
        self.red_lower2 = np.array([96, 31, 238])
        self.red_upper2 = np.array([21, 125, 195])
        
        # 轨迹追踪参数
        self.track_points = deque(maxlen=50)  # 保存最近50个位置点
        self.track_colors = deque(maxlen=50)  # 对应的颜色（用于渐变效果）
        
        # 形态学操作核
        self.kernel = np.ones((5, 5), np.uint8)
        
        # 最小检测面积
        self.min_area = 300
        
        # 圆形度阈值
        self.circularity_threshold = 0.6
        
    def get_red_mask(self, hsv_image):
        """
        获取红色掩码（红色在HSV中有两个范围）
        """
        mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        return mask
    
    def morphological_operations(self, mask):
        """
        形态学操作：腐蚀和膨胀，去除噪声并填补空洞
        """
        # 开运算：先腐蚀后膨胀，去除小噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # 闭运算：先膨胀后腐蚀，填补内部空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        # 额外的膨胀操作，增强目标
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        return mask
    
    def is_circular(self, contour):
        """
        判断轮廓是否接近圆形
        """
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return False
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        # 计算圆形度：4π*面积/周长²
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity > self.circularity_threshold
    
    def detect_red_cap(self, frame):
        """
        检测红色圆形瓶盖
        """
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 高斯模糊减少噪声
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # 获取红色掩码
        red_mask = self.get_red_mask(hsv)
        
        # 形态学操作
        processed_mask = self.morphological_operations(red_mask)
        
        # 查找轮廓
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cap = None
        best_area = 0
        
        # 寻找最佳的圆形红色目标
        for contour in contours:
            if self.is_circular(contour):
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
                    "position_pixels": [cx, cy]
                }
                
                # 在图像上绘制检测结果
                self.draw_detection(frame, best_cap, cx, cy)
        
        # 绘制轨迹
        self.draw_trajectory(frame)
        
        # 显示处理后的掩码（调试用）
        cv2.imshow('Red Mask', processed_mask)
        
        return result, frame
    
    def draw_detection(self, frame, contour, cx, cy):
        """
        在图像上绘制检测结果
        """
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
        cv2.putText(frame, f'Red Cap: ({cx}, {cy})', 
                   (cx - 80, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
    
    def draw_trajectory(self, frame):
        """
        绘制移动轨迹
        """
        if len(self.track_points) < 2:
            return
            
        # 绘制轨迹线
        for i in range(1, len(self.track_points)):
            # 计算颜色渐变（越新的点越亮）
            alpha = i / len(self.track_points)
            color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
            
            # 绘制连接线
            cv2.line(frame, self.track_points[i-1], self.track_points[i], color, 2)
            
            # 绘制轨迹点
            cv2.circle(frame, self.track_points[i], 3, color, -1)
        
        # 在左上角显示轨迹信息
        if len(self.track_points) > 0:
            current_pos = self.track_points[-1]
            cv2.putText(frame, f'Trajectory Points: {len(self.track_points)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f'Current: {current_pos}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def clear_trajectory(self):
        """
        清除轨迹
        """
        self.track_points.clear()

def main():
    """
    主函数
    """
    print("红色圆形瓶盖实时追踪系统启动中...")
    print("功能说明：")
    print("- 实时检测红色圆形瓶盖")
    print("- 显示瓶盖位置坐标")
    print("- 绘制移动轨迹")
    print("- 按 'c' 清除轨迹")
    print("- 按 'q' 退出程序")
    print("-" * 50)
    
    # 初始化追踪器
    tracker = RedCapTracker()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头数据")
                break
            
            # 检测红色瓶盖
            result, processed_frame = tracker.detect_red_cap(frame)
            
            # 输出检测结果
            if result:
                print(json.dumps(result, ensure_ascii=False))
            
            # 显示结果
            cv2.imshow('Red Cap Tracker', processed_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                tracker.clear_trajectory()
                print("轨迹已清除")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()