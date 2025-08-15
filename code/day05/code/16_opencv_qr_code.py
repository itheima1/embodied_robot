import cv2
from pyzbar import pyzbar
import numpy as np

def main():
    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，按 'q' 键退出程序")
    print("将摄像头对准二维码进行识别")
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查是否成功读取到画面
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 使用pyzbar检测和解码二维码
        barcodes = pyzbar.decode(frame)
        
        # 处理检测到的二维码
        for barcode in barcodes:
            # 提取二维码的边界框坐标
            (x, y, w, h) = barcode.rect
            
            # 在二维码周围绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 解码二维码数据
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type
            
            # 在图像上显示二维码信息
            text = f"Type: {barcode_type}"
            cv2.putText(frame, text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示二维码内容（如果内容太长，则截断显示）
            if len(barcode_data) > 50:
                display_data = barcode_data[:47] + "..."
            else:
                display_data = barcode_data
            
            cv2.putText(frame, f"Data: {display_data}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 在控制台打印完整的二维码信息
            print(f"检测到二维码 - 类型: {barcode_type}, 内容: {barcode_data}")
        
        # 在画面上显示提示信息
        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示画面
        cv2.imshow('QR Code Scanner', frame)
        
        # 检测按键，如果按下 'q' 键则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()