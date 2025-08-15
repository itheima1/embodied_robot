import cv2
import numpy as np
import json

def get_color_mask(hsv_image, color_name):
    """
    根据颜色名称获取HSV颜色掩码
    """
    if color_name == "red":
        # 红色在HSV中有两个范围（因为红色在色相环的两端）
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color_name == "black":
        # 黑色：低亮度值
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv_image, lower_black, upper_black)
    else:
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    return mask

def preprocess_image(frame):
    """
    图像预处理：转换到HSV空间
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 高斯模糊处理，减少噪声
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    return blurred

def detect_colored_objects(hsv_image, original_frame):
    """
    检测特定颜色的物体并返回结果
    """
    results = []
    colors_to_detect = ["red", "black"]
    
    for color_name in colors_to_detect:
        # 获取颜色掩码
        color_mask = get_color_mask(hsv_image, color_name)
        
        # 形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 过滤掉太小的轮廓
            area = cv2.contourArea(contour)
            if area < 500:  # 最小面积阈值
                continue
                
            # 计算轮廓的中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
                
            # 计算轮廓的近似多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 根据顶点数量判断形状
            vertices = len(approx)
            shape_type = "unknown"
            
            if vertices == 4:
                # 检查是否为正方形
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    shape_type = "square"
                else:
                    shape_type = "rectangle"
            elif vertices > 6:  # 降低顶点数要求
                 # 计算轮廓的圆形度
                 perimeter = cv2.arcLength(contour, True)
                 if perimeter > 0:
                     circularity = 4 * np.pi * area / (perimeter * perimeter)
                     if circularity > 0.5:  # 降低圆形度阈值，接近圆形也认为是圆形
                         shape_type = "circle"
            elif vertices == 3:
                shape_type = "triangle"
            
            # 在原图上绘制轮廓和标签
            if color_name == "red":
                color_bgr = (0, 0, 255)  # 红色
            elif color_name == "black":
                color_bgr = (255, 255, 255)  # 白色（用于在黑色物体上显示）
            else:
                color_bgr = (0, 255, 0)  # 绿色
                
            cv2.drawContours(original_frame, [contour], -1, color_bgr, 2)
            cv2.circle(original_frame, (cx, cy), 5, color_bgr, -1)
            
            # 显示颜色和形状信息
            label = f"{color_name}_{shape_type}"
            cv2.putText(original_frame, label, (cx-40, cy-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
            # 只添加圆形和正方形到结果列表
            if shape_type in ["circle", "square"]:
                 results.append({
                     "type": "shape",
                     "value": shape_type,
                     "position_pixels": [cx, cy]
                 })
        
        # 显示颜色掩码（用于调试）
        cv2.imshow(f'{color_name}_mask', color_mask)
    
    return results

def main():
    """
    主函数：形状分拣机
    """
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("形状分拣机已启动...")
    print("按 'q' 键退出程序")
    print("按 's' 键保存当前检测结果")
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头数据")
            break
        
        # 图像预处理
        hsv_image = preprocess_image(frame)
        
        # 检测彩色物体
        results = detect_colored_objects(hsv_image, frame)
        
        # 显示结果信息
        if results:
             result_text = json.dumps(results, ensure_ascii=False, indent=2)
             print(f"\n检测结果: {result_text}")
             shapes_info = [r['value'] for r in results]
             print(f"\r检测到 {len(results)} 个形状: {shapes_info}", end="")
        
        # 显示图像
        cv2.imshow('原图 - 物体检测', frame)
        cv2.imshow('HSV图像', hsv_image)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and results:
            # 保存检测结果到文件
            with open('shape_detection_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n检测结果已保存到 shape_detection_results.json")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")

if __name__ == "__main__":
    main()