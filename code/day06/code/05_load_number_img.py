import cv2
import numpy as np
import os

def load_and_process_image(image_path, target_size=(28, 28)):
    """
    加载图像并处理为指定尺寸的灰度图像
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (width, height)
    
    Returns:
        numpy.ndarray: 处理后的灰度图像数组
    """
    print(f"=== 图像处理程序 ===")
    print(f"目标图像: {image_path}")
    print(f"目标尺寸: {target_size[0]}x{target_size[1]} 像素")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件不存在 - {image_path}")
        return None
    
    try:
        # 读取图像
        print("正在读取图像...")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"错误：无法读取图像文件 - {image_path}")
            print("请检查文件格式是否正确")
            return None
        
        print(f"原始图像尺寸: {image.shape[1]}x{image.shape[0]} 像素")
        print(f"原始图像通道数: {image.shape[2]}")
        
        # 转换为灰度图像
        print("转换为灰度图像...")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"灰度图像尺寸: {gray_image.shape[1]}x{gray_image.shape[0]} 像素")
        print(f"灰度图像像素值范围: {gray_image.min()} - {gray_image.max()}")
        
        # 缩放到目标尺寸
        print(f"缩放图像到 {target_size[0]}x{target_size[1]} 像素...")
        resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
        
        print(f"缩放后图像尺寸: {resized_image.shape[1]}x{resized_image.shape[0]} 像素")
        print(f"缩放后像素值范围: {resized_image.min()} - {resized_image.max()}")
        
        return resized_image
        
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        return None

def enhance_contrast_filter(image_array, black_threshold=40, white_threshold=100):
    """
    对比度增强滤波算法
    
    Args:
        image_array: 输入图像数组
        black_threshold: 黑色阈值，低于此值的像素设为0
        white_threshold: 白色阈值，高于此值的像素按比例放大
    
    Returns:
        numpy.ndarray: 处理后的图像数组
    """
    if image_array is None:
        return None
    
    # 创建输出数组的副本
    filtered_array = image_array.copy().astype(np.float32)
    
    print(f"\n=== 对比度增强滤波 ===")
    print(f"黑色阈值: {black_threshold} (低于此值设为0)")
    print(f"白色阈值: {white_threshold} (高于此值按比例放大)")
    
    # 统计原始像素分布
    black_pixels_before = np.sum(image_array <= black_threshold)
    white_pixels_before = np.sum(image_array >= white_threshold)
    middle_pixels_before = np.sum((image_array > black_threshold) & (image_array < white_threshold))
    
    print(f"\n处理前像素分布:")
    print(f"  黑色区域 (≤{black_threshold}): {black_pixels_before} 个像素")
    print(f"  中间区域 ({black_threshold+1}-{white_threshold-1}): {middle_pixels_before} 个像素")
    print(f"  白色区域 (≥{white_threshold}): {white_pixels_before} 个像素")
    
    # 1. 将低于黑色阈值的像素设为0
    filtered_array[filtered_array <= black_threshold] = 0
    
    # 2. 对高于白色阈值的像素进行比例放大
    # 使用线性映射：将[white_threshold, 255]映射到[white_threshold, 255]
    # 但增加对比度，使用更陡峭的映射曲线
    white_mask = image_array >= white_threshold
    if np.any(white_mask):
        # 计算放大因子，确保最大值不超过255
        max_original = image_array[white_mask].max()
        if max_original > white_threshold:
            # 使用非线性映射增强对比度
            # 公式: new_value = white_threshold + (old_value - white_threshold) * scale_factor
            scale_factor = (255 - white_threshold) / (max_original - white_threshold)
            # 增加对比度，使用1.2倍的放大系数，但确保不超过255
            enhanced_scale = min(scale_factor * 1.2, (255 - white_threshold) / (max_original - white_threshold))
            
            filtered_array[white_mask] = white_threshold + (filtered_array[white_mask] - white_threshold) * enhanced_scale
    
    # 3. 中间值保持不变或轻微调整
    # 对中间值应用轻微的对比度增强
    middle_mask = (image_array > black_threshold) & (image_array < white_threshold)
    if np.any(middle_mask):
        # 对中间值应用S曲线增强对比度
        middle_values = filtered_array[middle_mask]
        # 归一化到0-1范围
        normalized = (middle_values - black_threshold) / (white_threshold - black_threshold)
        # 应用S曲线 (sigmoid-like)
        enhanced = 0.5 * (1 + np.tanh(4 * (normalized - 0.5)))
        # 映射回原始范围
        filtered_array[middle_mask] = black_threshold + enhanced * (white_threshold - black_threshold)
    
    # 确保所有值都在0-255范围内
    filtered_array = np.clip(filtered_array, 0, 255)
    
    # 转换回整数类型
    filtered_array = filtered_array.astype(np.uint8)
    
    # 统计处理后的像素分布
    black_pixels_after = np.sum(filtered_array == 0)
    white_pixels_after = np.sum(filtered_array >= white_threshold)
    middle_pixels_after = np.sum((filtered_array > 0) & (filtered_array < white_threshold))
    
    print(f"\n处理后像素分布:")
    print(f"  纯黑色 (=0): {black_pixels_after} 个像素")
    print(f"  中间区域 (1-{white_threshold-1}): {middle_pixels_after} 个像素")
    print(f"  白色区域 (≥{white_threshold}): {white_pixels_after} 个像素")
    
    print(f"\n滤波效果:")
    print(f"  原始像素值范围: {image_array.min()} - {image_array.max()}")
    print(f"  滤波后像素值范围: {filtered_array.min()} - {filtered_array.max()}")
    print(f"  对比度增强: {(filtered_array.max() - filtered_array.min()) / (image_array.max() - image_array.min()):.2f}x")
    
    return filtered_array

def print_pixel_array(image_array, title="像素数组"):
    """
    打印像素数组的详细信息
    
    Args:
        image_array: 图像数组
        title: 打印标题
    """
    if image_array is None:
        print("错误：图像数组为空")
        return
    
    print(f"\n=== {title} ===")
    print(f"数组形状: {image_array.shape}")
    print(f"数据类型: {image_array.dtype}")
    print(f"像素值范围: {image_array.min()} - {image_array.max()}")
    print(f"数组大小: {image_array.size} 个像素")
    
    print("\n完整像素数组:")
    print("(每行代表图像的一行，每个数字代表一个像素的灰度值)")
    print("(0=黑色, 255=白色)")
    print("-" * 60)
    
    # 打印数组，每行显示图像的一行
    for i, row in enumerate(image_array):
        print(f"第{i+1:2d}行: ", end="")
        for j, pixel in enumerate(row):
            print(f"{pixel:3d}", end=" ")
        print()  # 换行
    
    print("-" * 60)
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  平均像素值: {image_array.mean():.2f}")
    print(f"  标准差: {image_array.std():.2f}")
    print(f"  黑色像素(0-50): {np.sum((image_array >= 0) & (image_array <= 50))} 个")
    print(f"  灰色像素(51-200): {np.sum((image_array > 50) & (image_array <= 200))} 个")
    print(f"  白色像素(201-255): {np.sum((image_array > 200) & (image_array <= 255))} 个")

def save_array_to_file(image_array, filename="pixel_array.txt"):
    """
    将像素数组保存到文本文件
    
    Args:
        image_array: 图像数组
        filename: 保存的文件名
    """
    if image_array is None:
        print("错误：无法保存空数组")
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# 28x28像素灰度图像数组\n")
            f.write(f"# 图像尺寸: {image_array.shape[1]}x{image_array.shape[0]}\n")
            f.write(f"# 像素值范围: {image_array.min()} - {image_array.max()}\n")
            f.write(f"# 0=黑色, 255=白色\n")
            f.write("\n")
            
            # 写入数组数据
            for i, row in enumerate(image_array):
                f.write(f"# 第{i+1}行\n")
                for j, pixel in enumerate(row):
                    f.write(f"{pixel:3d}")
                    if j < len(row) - 1:
                        f.write(", ")
                f.write("\n")
            
            # 写入统计信息
            f.write(f"\n# 统计信息\n")
            f.write(f"# 平均像素值: {image_array.mean():.2f}\n")
            f.write(f"# 标准差: {image_array.std():.2f}\n")
            f.write(f"# 黑色像素(0-50): {np.sum((image_array >= 0) & (image_array <= 50))}\n")
            f.write(f"# 灰色像素(51-200): {np.sum((image_array > 50) & (image_array <= 200))}\n")
            f.write(f"# 白色像素(201-255): {np.sum((image_array > 200) & (image_array <= 255))}\n")
        
        print(f"\n像素数组已保存到文件: {filename}")
        
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

def visualize_processed_image(original_path, processed_array, filtered_array=None):
    """
    可视化处理前后的图像对比
    
    Args:
        original_path: 原始图像路径
        processed_array: 处理后的图像数组
        filtered_array: 滤波后的图像数组（可选）
    """
    if processed_array is None:
        print("无法显示图像：处理后的数组为空")
        return
    
    try:
        # 读取原始图像
        original = cv2.imread(original_path)
        if original is None:
            print("无法读取原始图像进行对比")
            return
        
        # 转换原始图像为灰度
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # 创建显示窗口
        print("\n显示图像对比 (按任意键关闭窗口)")
        
        # 显示原始图像
        cv2.imshow("1. Original Image", original_gray)
        
        # 显示处理后的图像
        cv2.imshow("2. Resized 28x28 Image", processed_array)
        
        # 放大显示处理后的图像以便观察
        enlarged = cv2.resize(processed_array, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("3. Resized 28x28 (Enlarged)", enlarged)
        
        # 如果有滤波后的图像，也显示出来
        if filtered_array is not None:
            cv2.imshow("4. Filtered 28x28 Image", filtered_array)
            
            # 放大显示滤波后的图像
            filtered_enlarged = cv2.resize(filtered_array, (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("5. Filtered 28x28 (Enlarged)", filtered_enlarged)
            
            # 创建对比图像
            comparison = np.hstack([enlarged, filtered_enlarged])
            cv2.imshow("6. Before vs After Filter Comparison", comparison)
        
        # 等待按键
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"显示图像时发生错误: {e}")

def main():
    """
    主函数
    """
    # 图像文件路径
    image_path = "captured_images/roi_20250729_144125.bmp"
    
    # 备选路径（如果图像在mnist目录下）
    alternative_path = "mnist/roi_20250729_144125.bmp"
    
    # 检查哪个路径存在
    if os.path.exists(image_path):
        target_path = image_path
    elif os.path.exists(alternative_path):
        target_path = alternative_path
    else:
        print("错误：找不到图像文件！")
        print(f"已检查路径:")
        print(f"  - {image_path}")
        print(f"  - {alternative_path}")
        print("\n请确保图像文件存在于以上路径之一")
        return
    
    # 处理图像
    processed_image = load_and_process_image(target_path, (28, 28))
    
    if processed_image is not None:
        # 打印原始像素数组
        print_pixel_array(processed_image, "28x28灰度像素数组（原始）")
        
        # 应用对比度增强滤波
        filtered_image = enhance_contrast_filter(processed_image, black_threshold=40, white_threshold=100)
        
        if filtered_image is not None:
            # 打印滤波后的像素数组
            print_pixel_array(filtered_image, "28x28灰度像素数组（滤波后）")
            
            # 保存原始和滤波后的数组到文件
            save_array_to_file(processed_image, "roi_28x28_pixel_array_original.txt")
            save_array_to_file(filtered_image, "roi_28x28_pixel_array_filtered.txt")
            
            # 可视化图像对比
            visualize_processed_image(target_path, processed_image, filtered_image)
            
            print("\n处理完成！")
            print("说明：")
            print("- 像素值0表示黑色")
            print("- 像素值255表示白色")
            print("- 中间值表示不同程度的灰色")
            print("- 原始数组已保存到 roi_28x28_pixel_array_original.txt")
            print("- 滤波后数组已保存到 roi_28x28_pixel_array_filtered.txt")
            print("\n滤波算法说明：")
            print("- 像素值 ≤ 40：设为0（纯黑色）")
            print("- 像素值 ≥ 100：按比例放大增强对比度")
            print("- 中间值：应用S曲线增强对比度")
        else:
            print("滤波处理失败！")
    else:
        print("图像处理失败！")

if __name__ == "__main__":
    main()