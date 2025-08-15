import cv2
import numpy as np
from ultralytics import YOLO
import os

def main():
    # 加载训练好的YOLO模型
    model_path = "YoLoDetect/best.pt"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    model = YOLO(model_path)
    print("模型加载成功！")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("摄像头已打开，按 'q' 键退出")
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 使用YOLO模型进行推理
        results = model(frame)
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 获取置信度
                    confidence = box.conf[0].cpu().numpy()
                    
                    # 获取类别
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # 只显示置信度大于0.5的检测结果
                    if confidence > 0.3:
                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 绘制标签和置信度
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # 绘制标签背景
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        
                        # 绘制标签文字
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 显示画面
        cv2.imshow('YOLO Detection', frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()