import cv2
import numpy as np
import os
from datetime import datetime

class ROISelector:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_selected = False
        self.current_frame = None
        self.original_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数，处理ROI选择
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制矩形
            self.drawing = True
            self.start_point = (x, y)
            self.roi_selected = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 如果正在绘制，更新矩形
            if self.drawing:
                self.current_frame = self.original_frame.copy()
                cv2.rectangle(self.current_frame, self.start_point, (x, y), (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 完成矩形绘制
            self.drawing = False
            self.end_point = (x, y)
            self.roi_selected = True
            cv2.rectangle(self.current_frame, self.start_point, self.end_point, (0, 255, 0), 2)
            
    def get_roi_coordinates(self):
        """
        获取ROI区域的坐标
        """
        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # 确保坐标顺序正确
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            
            return x_min, y_min, x_max, y_max
        return None
        
    def extract_roi(self, frame):
        """
        从图像中提取ROI区域
        """
        coords = self.get_roi_coordinates()
        if coords:
            x_min, y_min, x_max, y_max = coords
            roi = frame[y_min:y_max, x_min:x_max]
            return roi
        return None

def capture_and_select_roi():
    """
    主函数：打开摄像头，拍照并选择ROI
    """
    print("=== OpenCV 摄像头拍照和ROI选择程序 ===")
    print("操作说明：")
    print("1. 按 'c' 键拍照")
    print("2. 拍照后用鼠标拖拽选择感兴趣的区域")
    print("3. 按 's' 键保存选中的ROI区域")
    print("4. 按 'r' 键重新拍照")
    print("5. 按 'q' 键退出程序")
    print()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        print("请检查：")
        print("1. 摄像头是否正确连接")
        print("2. 摄像头是否被其他程序占用")
        print("3. 摄像头驱动是否正常")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("摄像头已成功打开！")
    print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    roi_selector = ROISelector()
    captured_frame = None
    mode = "preview"  # preview 或 select
    
    # 创建保存目录
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建保存目录: {save_dir}")
    
    while True:
        if mode == "preview":
            # 预览模式：显示实时摄像头画面
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break
                
            # 添加提示文字
            cv2.putText(frame, "Press 'c' to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Camera Preview", frame)
            
        elif mode == "select":
            # ROI选择模式：显示拍摄的照片并允许选择ROI
            if roi_selector.current_frame is not None:
                display_frame = roi_selector.current_frame.copy()
            else:
                display_frame = captured_frame.copy()
                
            # 添加提示文字
            cv2.putText(display_frame, "Drag to select ROI", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_frame, "Press 's' to save ROI", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_frame, "Press 'r' to recapture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow("Select ROI", display_frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("退出程序")
            break
            
        elif key == ord('c') and mode == "preview":
            # 拍照
            ret, frame = cap.read()
            if ret:
                captured_frame = frame.copy()
                roi_selector.original_frame = captured_frame.copy()
                roi_selector.current_frame = captured_frame.copy()
                
                # 切换到ROI选择模式
                mode = "select"
                cv2.destroyWindow("Camera Preview")
                cv2.namedWindow("Select ROI")
                cv2.setMouseCallback("Select ROI", roi_selector.mouse_callback)
                
                print("照片已拍摄！请用鼠标拖拽选择感兴趣的区域")
            else:
                print("拍照失败！")
                
        elif key == ord('r') and mode == "select":
            # 重新拍照
            mode = "preview"
            cv2.destroyWindow("Select ROI")
            cv2.namedWindow("Camera Preview")
            roi_selector = ROISelector()  # 重置ROI选择器
            print("返回预览模式，可以重新拍照")
            
        elif key == ord('s') and mode == "select" and roi_selector.roi_selected:
            # 保存ROI
            roi = roi_selector.extract_roi(captured_frame)
            if roi is not None and roi.size > 0:
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"roi_{timestamp}.bmp"
                filepath = os.path.join(save_dir, filename)
                
                # 保存ROI为BMP格式
                success = cv2.imwrite(filepath, roi)
                if success:
                    print(f"ROI区域已保存: {filepath}")
                    print(f"ROI尺寸: {roi.shape[1]}x{roi.shape[0]} 像素")
                    
                    # 显示保存的ROI
                    cv2.imshow("Saved ROI", roi)
                    print("按任意键关闭ROI预览窗口")
                    cv2.waitKey(0)
                    cv2.destroyWindow("Saved ROI")
                else:
                    print(f"保存失败: {filepath}")
            else:
                print("错误：ROI区域无效")
                
        elif key == ord('s') and mode == "select" and not roi_selector.roi_selected:
            print("请先用鼠标选择ROI区域！")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出，摄像头已关闭")

def main():
    """
    程序入口点
    """
    try:
        capture_and_select_roi()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()