import cv2
import numpy as np
import json
import math

def main():
    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开")
    print("按 's' 键拍摄背景图像")
    print("按 'q' 键退出程序")
    
    background = None
    background_captured = False
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查是否成功读取到画面
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 如果还没有拍摄背景图像
        if not background_captured:
            cv2.putText(frame, "Press 's' to capture background", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Object Pose Detection - Background Capture', frame)
        else:
            # 进行背景差分
            diff = cv2.absdiff(background, gray)
            
            # 应用阈值处理
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # 形态学操作去除噪声
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 存储检测到的物体信息
            detected_objects = []
            
            # 绘制轮廓和边界框
            for contour in contours:
                # 过滤小的轮廓
                area = cv2.contourArea(contour)
                if area > 500:  # 最小面积阈值
                    # 获取最小外接矩形（旋转矩形）
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    
                    # 提取矩形信息
                    center_x, center_y = rect[0]  # 中心点坐标
                    width, height = rect[1]       # 宽度和高度
                    angle = rect[2]               # 旋转角度
                    
                    # 角度标准化（OpenCV的角度范围是-90到0度）
                    # 将角度转换为0-360度范围
                    if width < height:
                        angle = angle + 90
                        width, height = height, width
                    
                    # 确保角度在0-360度范围内
                    if angle < 0:
                        angle = angle + 360
                    
                    # 绘制最小外接矩形
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                    
                    # 绘制中心点
                    center_x_int = int(center_x)
                    center_y_int = int(center_y)
                    cv2.circle(frame, (center_x_int, center_y_int), 5, (0, 0, 255), -1)
                    
                    # 绘制方向线（显示旋转角度）
                    length = 50
                    end_x = int(center_x + length * math.cos(math.radians(angle)))
                    end_y = int(center_y + length * math.sin(math.radians(angle)))
                    cv2.line(frame, (center_x_int, center_y_int), (end_x, end_y), (255, 0, 0), 2)
                    
                    # 添加到检测结果
                    detected_objects.append({
                        "type": "object",
                        "center_position": [float(center_x), float(center_y)],
                        "size": [float(width), float(height)],
                        "rotation_angle": float(angle),
                        "area": float(area)
                    })
                    
                    # 显示物体信息
                    info_text = f"Center: ({center_x_int}, {center_y_int})"
                    cv2.putText(frame, info_text, 
                               (center_x_int + 10, center_y_int - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    size_text = f"Size: {width:.1f}x{height:.1f}"
                    cv2.putText(frame, size_text, 
                               (center_x_int + 10, center_y_int - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    angle_text = f"Angle: {angle:.1f}°"
                    cv2.putText(frame, angle_text, 
                               (center_x_int + 10, center_y_int),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 输出JSON格式结果
            if detected_objects:
                json_output = json.dumps(detected_objects, ensure_ascii=False, indent=2)
                print(f"检测结果: {json_output}")
            
            # 显示状态信息
            cv2.putText(frame, f"Objects detected: {len(detected_objects)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to recapture background", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "Green: Bounding box, Red: Center, Blue: Direction", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示差分图像（可选）
            cv2.imshow('Difference', thresh)
            cv2.imshow('Object Pose Detection', frame)
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('s'):
            # 拍摄背景图像
            background = gray.copy()
            background_captured = True
            print("背景图像已拍摄，开始物体姿态检测")
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()