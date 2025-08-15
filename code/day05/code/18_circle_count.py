import cv2
import numpy as np
import json

def detect_circles(image, min_radius=10, max_radius=100):
    """
    使用霍夫圆变换检测圆形
    """
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    
    # 霍夫圆变换检测圆形
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # 圆心之间的最小距离
        param1=50,   # Canny边缘检测的高阈值
        param2=30,   # 累加器阈值
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    return circles

def is_circle_shape(contour):
    """
    判断轮廓是否接近圆形
    """
    # 计算轮廓面积
    area = cv2.contourArea(contour)
    if area < 100:  # 过滤太小的轮廓
        return False
    
    # 计算轮廓周长
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    # 计算圆形度 (4π*面积/周长²)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # 放宽圆形度条件，接近圆形即可
    return 0.5 < circularity < 1.5

def main():
    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("圆形零件计数程序已启动")
    print("按 's' 键拍摄背景图像")
    print("按 'q' 键退出程序")
    print("按 'r' 键重新拍摄背景")
    
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
            cv2.putText(frame, "Place parts on the surface first", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow('Circle Parts Counter - Background Capture', frame)
        else:
            # 进行背景差分
            diff = cv2.absdiff(background, gray)
            
            # 应用阈值处理
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # 轻度腐蚀操作，保持检测精度
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 存储检测到的圆形零件
            detected_circles = []
            circle_count = 0
            
            # 方法1：基于轮廓的圆形检测
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200 and is_circle_shape(contour):  # 面积阈值和圆形度检测
                    # 获取最小外接圆
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    # 过滤半径范围
                    if 10 < radius < 100:
                        # 绘制圆形
                        cv2.circle(frame, center, radius, (0, 255, 0), 2)
                        cv2.circle(frame, center, 2, (0, 0, 255), 3)
                        
                        # 添加编号
                        circle_count += 1
                        cv2.putText(frame, str(circle_count), 
                                   (center[0] - 10, center[1] - radius - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # 记录检测结果
                        detected_circles.append({
                            "id": int(circle_count),
                            "center": [int(center[0]), int(center[1])],
                            "radius": int(radius),
                            "area": float(area)
                        })
            
            # 方法2：霍夫圆变换检测（作为补充）
            circles = detect_circles(thresh)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # 检查是否与已检测的圆形重叠
                    is_duplicate = False
                    for detected in detected_circles:
                        dist = np.sqrt((x - detected["center"][0])**2 + (y - detected["center"][1])**2)
                        if dist < max(r, detected["radius"]) * 0.8:  # 重叠判断
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and 10 < r < 100:
                        # 绘制霍夫检测到的圆形（用不同颜色区分）
                        cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                        cv2.circle(frame, (x, y), 2, (255, 0, 255), 3)
                        
                        circle_count += 1
                        cv2.putText(frame, f"H{circle_count}", 
                                   (x - 15, y - r - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        
                        detected_circles.append({
                            "id": f"H{circle_count}",
                            "center": [int(x), int(y)],
                            "radius": int(r),
                            "method": "hough"
                        })
            
            # 输出检测结果
            if detected_circles:
                json_output = json.dumps(detected_circles, ensure_ascii=False, indent=2)
                print(f"\n检测到 {len(detected_circles)} 个圆形零件:")
                print(json_output)
            
            # 显示统计信息
            cv2.putText(frame, f"Circle Parts Count: {len(detected_circles)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'r' to recapture background", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "Green: Contour detection, Purple: Hough detection", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示处理后的图像
            cv2.imshow('Processed Image', thresh)
            cv2.imshow('Circle Parts Counter - Detection', frame)
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('s') or key == ord('r'):
            # 拍摄或重新拍摄背景图像
            background = gray.copy()
            background_captured = True
            action = "重新拍摄" if key == ord('r') else "拍摄"
            print(f"背景图像已{action}，开始圆形零件检测")
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()