import cv2

def main():
    # 加载Haar级联分类器用于人脸检测
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 检查分类器是否成功加载
    if face_cascade.empty():
        print("错误：无法加载人脸检测分类器")
        return
    
    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(1)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("人脸识别程序已启动，按 'q' 键退出程序")
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查是否成功读取到画面
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 转换为灰度图像（Haar分类器需要灰度图像）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # 图像缩放因子
            minNeighbors=5,       # 每个候选矩形应该保留的邻居数目
            minSize=(30, 30),     # 最小可能的人脸尺寸
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 在检测到的人脸周围绘制矩形框
        for (x, y, w, h) in faces:
            # 绘制绿色矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 在矩形框上方添加文字标签
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 在窗口左上角显示检测到的人脸数量
        face_count = len(faces)
        cv2.putText(frame, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 显示画面
        cv2.imshow('Face Detection', frame)
        
        # 检测按键，如果按下 'q' 键则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("人脸识别程序已退出")

if __name__ == "__main__":
    main()