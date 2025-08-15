import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

class HSVColorTuner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HSV颜色调节器 - 灰色检测")
        self.root.geometry("800x600")
        
        # 初始化HSV参数（灰色的初始值）
        self.h_min = tk.IntVar(value=0)
        self.s_min = tk.IntVar(value=0)
        self.v_min = tk.IntVar(value=20)
        self.h_max = tk.IntVar(value=180)
        self.s_max = tk.IntVar(value=50)
        self.v_max = tk.IntVar(value=220)
        
        # 摄像头相关
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        self.setup_ui()
        self.start_camera()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="HSV参数调节", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # HSV滑块
        self.create_slider(control_frame, "H最小值", self.h_min, 0, 180, 0)
        self.create_slider(control_frame, "S最小值", self.s_min, 0, 255, 1)
        self.create_slider(control_frame, "V最小值", self.v_min, 0, 255, 2)
        self.create_slider(control_frame, "H最大值", self.h_max, 0, 180, 3)
        self.create_slider(control_frame, "S最大值", self.s_max, 0, 255, 4)
        self.create_slider(control_frame, "V最大值", self.v_max, 0, 255, 5)
        
        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        # 重置按钮
        reset_btn = ttk.Button(button_frame, text="重置为默认值", command=self.reset_values)
        reset_btn.pack(pady=5)
        
        # 打印当前值按钮
        print_btn = ttk.Button(button_frame, text="打印当前HSV值", command=self.print_current_values)
        print_btn.pack(pady=5)
        
        # 右侧图像显示区域
        image_frame = ttk.LabelFrame(main_frame, text="实时图像", padding=10)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 原始图像
        self.original_label = ttk.Label(image_frame, text="原始图像")
        self.original_label.pack(pady=5)
        
        # 掩码图像
        self.mask_label = ttk.Label(image_frame, text="掩码图像")
        self.mask_label.pack(pady=5)
        
        # 结果图像
        self.result_label = ttk.Label(image_frame, text="检测结果")
        self.result_label.pack(pady=5)
        
    def create_slider(self, parent, label, variable, min_val, max_val, row):
        """创建滑块控件"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                          variable=variable, length=200)
        slider.grid(row=row, column=1, padx=10, pady=5)
        
        # 显示当前值的标签
        value_label = ttk.Label(parent, textvariable=variable)
        value_label.grid(row=row, column=2, padx=5, pady=5)
        
    def reset_values(self):
        """重置HSV值为默认值"""
        self.h_min.set(0)
        self.s_min.set(0)
        self.v_min.set(20)
        self.h_max.set(180)
        self.s_max.set(50)
        self.v_max.set(220)
        
    def print_current_values(self):
        """打印当前HSV值"""
        print("当前灰色HSV参数:")
        print(f"lower: np.array([{self.h_min.get()}, {self.s_min.get()}, {self.v_min.get()}])")
        print(f"upper: np.array([{self.h_max.get()}, {self.s_max.get()}, {self.v_max.get()}])")
        print("-" * 50)
        
    def start_camera(self):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            return
            
        self.is_running = True
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
    def camera_loop(self):
        """摄像头循环"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.process_frame(frame)
            time.sleep(0.03)  # 约30fps
            
    def process_frame(self, frame):
        """处理帧"""
        # 调整图像大小
        frame = cv2.resize(frame, (320, 240))
        
        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 获取当前HSV参数
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
        upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 应用掩码到原图
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 查找轮廓并绘制
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                # 计算中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 绘制中心点和轮廓
                    cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                    cv2.putText(result, f'Gray({cx},{cy})', 
                              (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.4, (255, 255, 255), 1)
        
        # 更新UI显示
        self.update_image_display(frame, mask, result)
        
    def update_image_display(self, original, mask, result):
        """更新图像显示"""
        try:
            # 转换原始图像
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_rgb)
            original_tk = ImageTk.PhotoImage(original_pil)
            
            # 转换掩码图像
            mask_pil = Image.fromarray(mask)
            mask_tk = ImageTk.PhotoImage(mask_pil)
            
            # 转换结果图像
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            result_tk = ImageTk.PhotoImage(result_pil)
            
            # 更新标签
            self.original_label.configure(image=original_tk)
            self.original_label.image = original_tk
            
            self.mask_label.configure(image=mask_tk)
            self.mask_label.image = mask_tk
            
            self.result_label.configure(image=result_tk)
            self.result_label.image = result_tk
            
        except Exception as e:
            print(f"更新图像显示时出错: {e}")
            
    def on_closing(self):
        """关闭程序时的清理工作"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        
    def run(self):
        """运行程序"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """主函数"""
    print("HSV颜色调节器启动中...")
    print("使用滑块调节HSV参数以获得最佳的灰色检测效果")
    print("调节完成后点击'打印当前HSV值'按钮获取参数")
    print("-" * 50)
    
    tuner = HSVColorTuner()
    tuner.run()

if __name__ == "__main__":
    main()