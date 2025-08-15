import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

class RedHSVColorTuner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HSV颜色调节器 - 红色瓶盖检测")
        self.root.geometry("800x600")
        
        # 简化的HSV参数（红色瓶盖优化）
        self.h_min = tk.IntVar(value=0)
        self.s_min = tk.IntVar(value=100)
        self.v_min = tk.IntVar(value=100)
        self.h_max = tk.IntVar(value=10)
        self.s_max = tk.IntVar(value=255)
        self.v_max = tk.IntVar(value=255)
        
        # 预设HSV值列表
        self.presets = {
            "红色瓶盖1": [0, 100, 100, 10, 255, 255],
            "红色瓶盖2": [0, 120, 70, 10, 255, 255],
            "鲜红色": [0, 150, 150, 8, 255, 255],
            "深红色": [170, 100, 50, 180, 255, 200],
            "橙红色": [5, 100, 100, 15, 255, 255]
        }
        
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
        
        # 预设值按钮框架
        preset_frame = ttk.LabelFrame(control_frame, text="快速预设", padding=5)
        preset_frame.pack(fill=tk.X, pady=5)
        
        # 创建预设按钮
        for i, (name, values) in enumerate(self.presets.items()):
            btn = ttk.Button(preset_frame, text=name, 
                           command=lambda v=values: self.apply_preset(v))
            btn.pack(fill=tk.X, pady=2)
        
        # HSV滑块框架
        slider_frame = ttk.LabelFrame(control_frame, text="红色HSV调节", padding=5)
        slider_frame.pack(fill=tk.X, pady=5)
        
        self.create_slider(slider_frame, "H最小值", self.h_min, 0, 180, 0)
        self.create_slider(slider_frame, "S最小值", self.s_min, 0, 255, 1)
        self.create_slider(slider_frame, "V最小值", self.v_min, 0, 255, 2)
        self.create_slider(slider_frame, "H最大值", self.h_max, 0, 180, 3)
        self.create_slider(slider_frame, "S最大值", self.s_max, 0, 255, 4)
        self.create_slider(slider_frame, "V最大值", self.v_max, 0, 255, 5)
        
        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        # 重置按钮
        reset_btn = ttk.Button(button_frame, text="重置为默认值", command=self.reset_values)
        reset_btn.pack(pady=5, fill=tk.X)
        
        # 打印当前值按钮
        print_btn = ttk.Button(button_frame, text="打印当前HSV值", command=self.print_current_values)
        print_btn.pack(pady=5, fill=tk.X)
        
        # 保存参数按钮
        save_btn = ttk.Button(button_frame, text="保存参数", command=self.save_current_values)
        save_btn.pack(pady=5, fill=tk.X)
        
        # 说明文本
        info_text = tk.Text(button_frame, height=8, width=30, wrap=tk.WORD)
        info_text.pack(pady=5, fill=tk.BOTH, expand=True)
        info_text.insert(tk.END, 
            "简化版HSV调节器:\n"
            "1. 点击预设按钮快速设置参数\n"
            "2. 微调HSV滑块优化检测效果\n"
            "3. 观察右侧实时检测结果\n"
            "4. 找到满意参数后点击'保存'\n"
            "5. 使用'打印参数'查看代码\n"
            "6. 按ESC或点击'退出'关闭\n\n"
            "✓ 已优化: 适应镜头畸变\n"
            "✓ 支持: 圆形和近似正方形\n"
            "提示: 先试试预设值!"
        )
        info_text.config(state=tk.DISABLED)
        
        # 右侧图像显示区域
        image_frame = ttk.LabelFrame(main_frame, text="实时图像", padding=10)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 原始图像
        self.original_label = ttk.Label(image_frame, text="原始图像")
        self.original_label.pack(pady=5)
        
        # 掩码图像
        self.mask_label = ttk.Label(image_frame, text="红色掩码")
        self.mask_label.pack(pady=5)
        
        # 结果图像
        self.result_label = ttk.Label(image_frame, text="检测结果")
        self.result_label.pack(pady=5)
        
    def apply_preset(self, values):
        """应用预设HSV值"""
        self.h_min.set(values[0])
        self.s_min.set(values[1])
        self.v_min.set(values[2])
        self.h_max.set(values[3])
        self.s_max.set(values[4])
        self.v_max.set(values[5])
        print(f"已应用预设值: H({values[0]}-{values[3]}) S({values[1]}-{values[4]}) V({values[2]}-{values[5]})")
    
    def create_slider(self, parent, label, variable, min_val, max_val, row):
        """创建滑块控件"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                          variable=variable, length=150)
        slider.grid(row=row, column=1, padx=5, pady=2)
        
        # 显示当前值的标签
        value_label = ttk.Label(parent, textvariable=variable)
        value_label.grid(row=row, column=2, padx=5, pady=2)
        
    def reset_values(self):
        """重置HSV值为默认值"""
        self.h_min.set(0)
        self.s_min.set(100)
        self.v_min.set(100)
        self.h_max.set(10)
        self.s_max.set(255)
        self.v_max.set(255)
        
    def print_current_values(self):
        """打印当前HSV值"""
        print("当前红色HSV参数:")
        print(f"red_lower = np.array([{self.h_min.get()}, {self.s_min.get()}, {self.v_min.get()}])")
        print(f"red_upper = np.array([{self.h_max.get()}, {self.s_max.get()}, {self.v_max.get()}])")
        print("-" * 60)
    
    def save_current_values(self):
        """保存当前HSV值到文件"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"red_hsv_params_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 红色瓶盖HSV参数\n")
            f.write(f"# 保存时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("import numpy as np\n\n")
            f.write(f"red_lower = np.array([{self.h_min.get()}, {self.s_min.get()}, {self.v_min.get()}])\n")
            f.write(f"red_upper = np.array([{self.h_max.get()}, {self.s_max.get()}, {self.v_max.get()}])\n")
        
        print(f"参数已保存到文件: {filename}")
        self.result_label.config(text=f"参数已保存: {filename}")
        
    def start_camera(self):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(2)
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
        
        # 高斯模糊减少噪声
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # 获取当前HSV参数
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
        upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 应用掩码到原图
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 查找轮廓并绘制
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        red_objects_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # 最小面积阈值
                # 计算中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 计算圆形度
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # 绘制中心点和轮廓
                        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                        
                        # 显示信息 - 降低圆形度阈值，适应镜头畸变
                        if circularity > 0.4:  # 降低阈值，接近正方形也认为是有效目标
                            shape_type = "瓶盖"
                        else:
                            shape_type = "其他"
                        cv2.putText(result, f'Red {shape_type}({cx},{cy})', 
                                  (cx-60, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.4, (255, 255, 255), 1)
                        cv2.putText(result, f'Area:{int(area)}', 
                                  (cx-40, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.3, (255, 255, 255), 1)
                        
                        red_objects_count += 1
        
        # 在图像上显示检测到的红色物体数量
        cv2.putText(result, f'Red Objects: {red_objects_count}', 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
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
    print("红色瓶盖HSV颜色调节器启动中...")
    print("使用滑块调节HSV参数以获得最佳的红色检测效果")
    print("红色在HSV色彩空间中分布在两个范围:")
    print("- 范围1: 0-10度 (纯红色)")
    print("- 范围2: 170-180度 (深红色)")
    print("调节完成后点击'打印当前HSV值'按钮获取参数")
    print("-" * 60)
    
    tuner = RedHSVColorTuner()
    tuner.run()

if __name__ == "__main__":
    main()