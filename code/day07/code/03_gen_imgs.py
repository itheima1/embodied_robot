import cv2
import numpy as np
import random
import os

def create_random_image_with_overlay():
    """创建带有随机纹理背景和叠加PNG图像的合成图片"""
    # 创建640x640的纹理背景图片
    background = create_textured_background(640, 640)
    
    # 随机选择一个PNG文件
    png_files = ['YoLoDetect/gaizi.png', 'YoLoDetect/gaizi2.png']
    selected_png = random.choice(png_files)
    
    # 检查文件是否存在
    if not os.path.exists(selected_png):
        print(f"警告: 文件 {selected_png} 不存在")
        return background, None
    
    # 读取PNG图片（包含透明通道）
    overlay = cv2.imread(selected_png, cv2.IMREAD_UNCHANGED)
    
    if overlay is None:
        print(f"错误: 无法读取文件 {selected_png}")
        return background, None
    
    # 如果PNG没有透明通道，添加一个全不透明的alpha通道
    if overlay.shape[2] == 3:
        alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
        overlay = np.concatenate([overlay, alpha], axis=2)
    
    # 先裁剪掉透明部分
    overlay = crop_transparent_parts(overlay)
    
    # 对裁剪后的overlay进行随机变换
    overlay = apply_random_transform(overlay)
    
    # 获取变换后overlay图片的尺寸
    overlay_h, overlay_w = overlay.shape[:2]
    
    # 计算随机放置位置
    max_x = max(0, 640 - overlay_w)
    max_y = max(0, 640 - overlay_h)
    
    x_offset = random.randint(0, max_x) if max_x > 0 else 0
    y_offset = random.randint(0, max_y) if max_y > 0 else 0
    
    # 确保overlay不会超出背景图片边界
    if x_offset < 0 or y_offset < 0 or \
       x_offset + overlay_w > 640 or y_offset + overlay_h > 640:
        # 如果overlay太大，需要缩放
        max_width = 640 - 100  # 留出边距
        max_height = 300  # 上方区域最大高度
        
        scale_w = max_width / overlay_w
        scale_h = max_height / overlay_h
        scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
        
        new_w = int(overlay_w * scale)
        new_h = int(overlay_h * scale)
        
        overlay = cv2.resize(overlay, (new_w, new_h))
        overlay_h, overlay_w = overlay.shape[:2]
        
        # 重新计算位置
        x_offset = (640 - overlay_w) // 2
        y_offset = 50
    
    # 提取alpha通道
    alpha = overlay[:, :, 3] / 255.0
    
    # 计算实际的非透明区域边界框
    actual_bbox = get_actual_bbox(overlay, x_offset, y_offset)
    
    # 获取要覆盖的区域
    roi = background[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w]
    
    # 进行alpha混合
    for c in range(3):  # BGR三个通道
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay[:, :, c]
    
    # 将混合后的区域放回背景图片
    background[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = roi
    
    # 使用实际边界框计算YOLO格式的位置信息
    if actual_bbox is not None:
        bbox_x, bbox_y, bbox_w, bbox_h = actual_bbox
        yolo_info = calculate_yolo_format(bbox_x, bbox_y, bbox_w, bbox_h, 640, 640)
    else:
        # 如果没有找到非透明区域，使用原始方法
        yolo_info = calculate_yolo_format(x_offset, y_offset, overlay_w, overlay_h, 640, 640)
    
    return background, yolo_info

def create_textured_background(width, height):
    """创建带纹理的背景图片"""
    # 随机选择背景类型
    bg_type = random.choice(['stripes', 'gradient', 'noise', 'checkerboard'])
    
    if bg_type == 'stripes':
        # 条纹背景
        background = np.zeros((height, width, 3), dtype=np.uint8)
        stripe_width = random.randint(10, 50)
        color1 = [random.randint(50, 200) for _ in range(3)]
        color2 = [random.randint(50, 200) for _ in range(3)]
        
        for i in range(0, width, stripe_width * 2):
            background[:, i:i+stripe_width] = color1
            if i + stripe_width < width:
                background[:, i+stripe_width:i+stripe_width*2] = color2
                
    elif bg_type == 'gradient':
        # 渐变背景
        background = np.zeros((height, width, 3), dtype=np.uint8)
        color1 = np.array([random.randint(0, 255) for _ in range(3)])
        color2 = np.array([random.randint(0, 255) for _ in range(3)])
        
        for i in range(height):
            ratio = i / height
            color = color1 * (1 - ratio) + color2 * ratio
            background[i, :] = color.astype(np.uint8)
            
    elif bg_type == 'noise':
        # 噪声背景
        base_color = [random.randint(100, 150) for _ in range(3)]
        noise = np.random.randint(-50, 50, (height, width, 3))
        background = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
        
    else:  # checkerboard
        # 棋盘背景
        background = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = random.randint(20, 80)
        color1 = [random.randint(50, 150) for _ in range(3)]
        color2 = [random.randint(100, 200) for _ in range(3)]
        
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    background[i:i+square_size, j:j+square_size] = color1
                else:
                    background[i:i+square_size, j:j+square_size] = color2
    
    return background

def calculate_yolo_format(x_offset, y_offset, width, height, img_width, img_height):
    """计算YOLO格式的位置信息"""
    # 计算中心点坐标
    center_x = x_offset + width / 2
    center_y = y_offset + height / 2
    
    # 转换为相对坐标（0-1之间）
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return center_x_norm, center_y_norm, width_norm, height_norm

def get_actual_bbox(overlay, x_offset, y_offset):
    """计算实际非透明区域的边界框"""
    # 获取alpha通道
    alpha = overlay[:, :, 3]
    
    # 找到非透明像素的位置
    non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) == 0:
        return None  # 没有非透明像素
    
    # 计算边界框
    min_y, max_y = np.min(non_transparent[0]), np.max(non_transparent[0])
    min_x, max_x = np.min(non_transparent[1]), np.max(non_transparent[1])
    
    # 转换为全局坐标
    global_x = x_offset + min_x
    global_y = y_offset + min_y
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    
    return global_x, global_y, bbox_w, bbox_h

def crop_transparent_parts(image):
    """裁剪掉图像的透明部分"""
    # 获取alpha通道
    alpha = image[:, :, 3]
    
    # 找到非透明像素的位置
    non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) == 0:
        # 如果没有非透明像素，返回一个最小的图像
        return np.zeros((1, 1, 4), dtype=image.dtype)
    
    # 计算边界框
    min_y, max_y = np.min(non_transparent[0]), np.max(non_transparent[0])
    min_x, max_x = np.min(non_transparent[1]), np.max(non_transparent[1])
    
    # 裁剪图像
    cropped = image[min_y:max_y+1, min_x:max_x+1]
    
    return cropped

def apply_random_transform(image):
    """对图像应用随机变换"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 随机旋转角度（-15到15度，避免过度旋转）
    angle = random.uniform(-15, 15)
    
    # 随机缩放（0.8到1.2倍）
    scale = random.uniform(0.8, 1.2)
    
    # 创建旋转和缩放矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 计算旋转后图像的边界框
    cos_a = abs(rotation_matrix[0, 0])
    sin_a = abs(rotation_matrix[0, 1])
    
    # 计算新的图像尺寸
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # 调整旋转矩阵的平移部分，使图像居中
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # 随机仿射变换（轻微的透视变化）
    dx = random.uniform(-10, 10)
    dy = random.uniform(-10, 10)
    rotation_matrix[0, 2] += dx
    rotation_matrix[1, 2] += dy
    
    # 应用变换，使用新的画布尺寸
    transformed = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    return transformed

def generate_dataset(num_images=1001):
    """生成数据集"""
    # 创建输出目录
    images_dir = "generated_dataset/images"
    labels_dir = "generated_dataset/labels"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"开始生成 {num_images} 张图片和标注文件...")
    
    for i in range(num_images):
        # 生成文件名（0000.jpg 到 1000.jpg）
        filename = f"{i:04d}"
        img_path = os.path.join(images_dir, f"{filename}.jpg")
        label_path = os.path.join(labels_dir, f"{filename}.txt")
        
        # 生成图片和标注信息
        image, yolo_info = create_random_image_with_overlay()
        
        # 保存图片
        cv2.imwrite(img_path, image)
        
        # 保存标注文件
        if yolo_info is not None:
            class_id = 0  # gaizi.png的类别编号
            center_x, center_y, width, height = yolo_info
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        else:
            # 如果没有标注信息，创建空文件
            with open(label_path, 'w') as f:
                pass
        
        # 显示进度
        if (i + 1) % 100 == 0 or i == 0:
            print(f"已生成 {i + 1}/{num_images} 张图片")
    
    print(f"数据集生成完成！")
    print(f"图片保存在: {images_dir}")
    print(f"标注保存在: {labels_dir}")

if __name__ == "__main__":
    # 生成数据集（0000.jpg 到 1000.jpg，共1001张图片）
    generate_dataset(1001)