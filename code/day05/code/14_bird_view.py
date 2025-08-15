import cv2
import numpy as np

class BirdViewTransform:
    def __init__(self):
        self.points = []  # 存储用户点击的四个点
        self.image = None
        self.original_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于获取用户点击的四个顶点"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                print(f"选择了第{len(self.points)}个点: ({x}, {y})")
                
                # 在图像上标记点击的点
                cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.image, str(len(self.points)), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Original Image - Click 4 corners', self.image)
                
                # 如果已经选择了4个点，进行透视变换
                if len(self.points) == 4:
                    self.perform_perspective_transform()
    
    def perform_perspective_transform(self):
        """执行透视变换"""
        # 将点转换为numpy数组
        src_points = np.float32(self.points)
        
        # 计算目标矩形的尺寸
        # 使用选择的四个点来估算合适的输出尺寸
        width = max(
            np.linalg.norm(src_points[0] - src_points[1]),
            np.linalg.norm(src_points[2] - src_points[3])
        )
        height = max(
            np.linalg.norm(src_points[0] - src_points[3]),
            np.linalg.norm(src_points[1] - src_points[2])
        )
        
        # 定义目标矩形的四个顶点（俯视图）
        dst_points = np.float32([
            [0, 0],                    # 左上角
            [width, 0],                # 右上角
            [width, height],           # 右下角
            [0, height]                # 左下角
        ])
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        bird_view = cv2.warpPerspective(self.original_image, matrix, (int(width), int(height)))
        
        # 显示结果
        cv2.imshow('Bird View (Press any key to reset)', bird_view)
        print("透视变换完成！按任意键重置选择点。")
        
        # 等待按键，然后重置
        cv2.waitKey(0)
        self.reset_points()
    
    def reset_points(self):
        """重置选择的点"""
        self.points = []
        self.image = self.original_image.copy()
        cv2.imshow('Original Image - Click 4 corners', self.image)
        print("已重置，请重新选择四个顶点")
    
    def run(self, image_path=None):
        """运行鸟瞰图变换程序"""
        if image_path:
            # 从文件加载图像
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                print(f"无法加载图像: {image_path}")
                return
        else:
            # 使用摄像头
            cap = cv2.VideoCapture(2)
            if not cap.isOpened():
                print("无法打开摄像头")
                return
            
            print("按空格键捕获当前帧，按ESC退出")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow('Camera - Press SPACE to capture, ESC to exit', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC键退出
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == 32:  # 空格键捕获
                    self.original_image = frame.copy()
                    cap.release()
                    break
        
        # 创建工作副本
        self.image = self.original_image.copy()
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow('Original Image - Click 4 corners')
        cv2.setMouseCallback('Original Image - Click 4 corners', self.mouse_callback)
        
        # 显示图像
        cv2.imshow('Original Image - Click 4 corners', self.image)
        
        print("使用说明：")
        print("1. 按顺序点击要矫正区域的四个顶点（建议按左上、右上、右下、左下的顺序）")
        print("2. 点击完四个点后会自动生成鸟瞰图")
        print("3. 在鸟瞰图窗口按任意键可以重新选择点")
        print("4. 按ESC键退出程序")
        
        # 主循环
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                break
            elif key == ord('r'):  # R键重置
                self.reset_points()
        
        cv2.destroyAllWindows()

def main():
    """主函数"""
    transform = BirdViewTransform()
    
    # 可以选择使用图像文件或摄像头
    print("选择输入方式：")
    print("1. 使用摄像头（默认）")
    print("2. 使用图像文件")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == '2':
        image_path = input("请输入图像文件路径: ").strip()
        transform.run(image_path)
    else:
        transform.run()

if __name__ == "__main__":
    main()