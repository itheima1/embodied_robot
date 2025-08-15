import cv2
import numpy as np

class ROISelector:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_selected = False
        self.roi_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于选择ROI区域"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 鼠标左键按下，开始绘制
            self.drawing = True
            self.start_point = (x, y)
            self.roi_selected = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 鼠标移动，实时更新矩形
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 鼠标左键释放，完成选择
            self.drawing = False
            self.end_point = (x, y)
            self.roi_selected = True
            
    def get_roi_coordinates(self):
        """获取ROI区域的坐标"""
        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            # 确保坐标顺序正确
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            return x_min, y_min, x_max, y_max
        return None
        
    def extract_roi(self, frame):
        """从原始帧中提取ROI区域"""
        coords = self.get_roi_coordinates()
        if coords:
            x_min, y_min, x_max, y_max = coords
            # 确保坐标在图像范围内
            h, w = frame.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max > x_min and y_max > y_min:
                return frame[y_min:y_max, x_min:x_max]
        return None

def main():
    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开")
    print("使用说明：")
    print("1. 在摄像头窗口中用鼠标拖拽选择ROI区域")
    print("2. 选择完成后，ROI区域会在新窗口中显示")
    print("3. 按 'r' 键重新选择ROI区域")
    print("4. 按 'q' 键退出程序")
    
    # 创建ROI选择器
    roi_selector = ROISelector()
    
    # 创建主窗口并设置鼠标回调
    cv2.namedWindow('Camera - Select ROI', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Camera - Select ROI', roi_selector.mouse_callback)
    
    roi_window_created = False
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查是否成功读取到画面
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 创建显示帧的副本
        display_frame = frame.copy()
        
        # 如果正在绘制或已选择ROI，绘制矩形
        if roi_selector.start_point and roi_selector.end_point:
            cv2.rectangle(display_frame, roi_selector.start_point, 
                         roi_selector.end_point, (0, 255, 0), 2)
        
        # 显示主画面
        cv2.imshow('Camera - Select ROI', display_frame)
        
        # 如果ROI已选择，提取并显示ROI区域
        if roi_selector.roi_selected:
            roi_frame = roi_selector.extract_roi(frame)
            if roi_frame is not None and roi_frame.size > 0:
                # 如果ROI太小，放大显示
                h, w = roi_frame.shape[:2]
                if h < 100 or w < 100:
                    scale_factor = max(100/h, 100/w)
                    new_h = int(h * scale_factor)
                    new_w = int(w * scale_factor)
                    roi_frame = cv2.resize(roi_frame, (new_w, new_h), 
                                         interpolation=cv2.INTER_LINEAR)
                
                # 创建ROI窗口（如果还没创建）
                if not roi_window_created:
                    cv2.namedWindow('ROI Region', cv2.WINDOW_AUTOSIZE)
                    roi_window_created = True
                
                cv2.imshow('ROI Region', roi_frame)
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('r'):
            print("重新选择ROI区域")
            roi_selector.roi_selected = False
            roi_selector.start_point = None
            roi_selector.end_point = None
            # 关闭ROI窗口
            if roi_window_created:
                cv2.destroyWindow('ROI Region')
                roi_window_created = False
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()