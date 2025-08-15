import cv2
import numpy as np
import json

class ColorSorter:
    def __init__(self):
        # 定义HSV颜色范围
        self.color_ranges = {
            'orange': {
                'lower': np.array([5, 50, 50]),
                'upper': np.array([35, 255, 255])
            },
            'blue': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'black': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 30])
            },
            'gray': {
                'lower': np.array([0, 0, 20]),
                'upper': np.array([180, 50, 220])
            }
        }
        
    def detect_colors(self, frame):
        """检测帧中的颜色并返回位置信息"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        results = []
        
        for color_name, color_range in self.color_ranges.items():
            # 创建颜色掩码
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # 形态学操作去除噪声
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 过滤小的轮廓
                if cv2.contourArea(contour) > 500:
                    # 计算轮廓的中心点
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 添加到结果中
                        results.append({
                            "type": "color",
                            "value": color_name,
                            "position_pixels": [cx, cy]
                        })
                        
                        # 在图像上绘制检测结果
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        cv2.putText(frame, f'{color_name}({cx},{cy})', 
                                  (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (255, 255, 255), 2)
                        
                        # 绘制轮廓
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        return results, frame
    
    def run(self):
        """运行颜色分拣器"""
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        print("智能颜色分拣机已启动")
        print("按 'q' 键退出程序")
        print("检测颜色：橘黄色、蓝色、黑色、灰色")
        print("-" * 50)
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头数据")
                break
            
            # 检测颜色
            results, processed_frame = self.detect_colors(frame)
            
            # 输出检测结果
            if results:
                print(f"检测结果: {json.dumps(results, ensure_ascii=False)}")
            
            # 显示处理后的图像
            cv2.imshow('智能颜色分拣机', processed_frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

def main():
    """主函数"""
    sorter = ColorSorter()
    sorter.run()

if __name__ == "__main__":
    main()