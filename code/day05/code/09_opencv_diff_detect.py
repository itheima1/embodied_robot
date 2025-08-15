import cv2
import numpy as np
import json

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
            cv2.imshow('Table Cleaner - Background Capture', frame)
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
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 计算中心点
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 绘制中心点
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # 添加到检测结果
                    detected_objects.append({
                        "type": "object",
                        "position_pixels": [center_x, center_y]
                    })
                    
                    # 显示坐标信息
                    cv2.putText(frame, f"({center_x}, {center_y})", 
                               (center_x + 10, center_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 输出JSON格式结果
            if detected_objects:
                json_output = json.dumps(detected_objects, ensure_ascii=False)
                print(f"检测结果: {json_output}")
            
            # 显示状态信息
            cv2.putText(frame, f"Objects detected: {len(detected_objects)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to recapture background", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 显示差分图像（可选）
            cv2.imshow('Difference', thresh)
            cv2.imshow('Table Cleaner - Object Detection', frame)
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('s'):
            # 拍摄背景图像
            background = gray.copy()
            background_captured = True
            print("背景图像已拍摄，开始物体检测")
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()