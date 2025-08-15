import cv2
import torch
from ultralytics import YOLO
import numpy as np

def main():
    # 加载训练好的YOLO模型
    model_path = r'e:\jszn_bj\day06\code\yolo\MaskDetector\best.pt'
    model = YOLO(model_path)
    
    # 强制使用CPU进行推理
    model.to('cpu')
    
    # 打开摄像头
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率 - 提高分辨率以获得更好的检测效果
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 设置摄像头其他参数
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
    
    # 定义类别名称和颜色
    class_names = ['mask', 'no-mask']
    colors = [(0, 255, 0), (0, 0, 255)]  # 绿色表示戴口罩，红色表示未戴口罩
    
    print("按 'q' 键退出程序")
    
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 图像预处理 - 提高图像质量
        # 调整亮度和对比度
        enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        # 降噪处理
        denoised_frame = cv2.bilateralFilter(enhanced_frame, 9, 75, 75)
        
        # 使用YOLO模型进行推理 - 添加更多参数优化
        results = model(denoised_frame, conf=0.3, iou=0.5, imgsz=640)
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 获取置信度和类别
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 只显示置信度大于0.3的检测结果
                    if confidence > 0.3:
                        # 获取类别名称和颜色
                        class_name = class_names[class_id]
                        color = colors[class_id]
                        
                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 绘制标签和置信度
                        label = f'{class_name}: {confidence:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # 绘制标签背景
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # 绘制标签文字
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 在画面上显示说明信息
        cv2.putText(frame, "Mask Detection - Press 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示画面
        cv2.imshow('Mask Detection', frame)
        
        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()