import cv2
import numpy as np
import os
from joblib import load
import matplotlib.pyplot as plt
import matplotlib

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_image(image_path, target_size=(28, 28)):
    """
    加载图像并处理为指定尺寸的灰度图像
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (width, height)
    
    Returns:
        numpy.ndarray: 处理后的灰度图像数组
    """
    print(f"=== 图像预处理 ===")
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
        
        # 转换为灰度图像
        print("转换为灰度图像...")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"灰度图像尺寸: {gray_image.shape[1]}x{gray_image.shape[0]} 像素")
        
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
    
    # 1. 将低于黑色阈值的像素设为0
    filtered_array[filtered_array <= black_threshold] = 0
    
    # 2. 对高于白色阈值的像素进行比例放大
    white_mask = image_array >= white_threshold
    if np.any(white_mask):
        max_original = image_array[white_mask].max()
        if max_original > white_threshold:
            scale_factor = (255 - white_threshold) / (max_original - white_threshold)
            enhanced_scale = min(scale_factor * 1.2, (255 - white_threshold) / (max_original - white_threshold))
            filtered_array[white_mask] = white_threshold + (filtered_array[white_mask] - white_threshold) * enhanced_scale
    
    # 3. 中间值应用S曲线增强对比度
    middle_mask = (image_array > black_threshold) & (image_array < white_threshold)
    if np.any(middle_mask):
        middle_values = filtered_array[middle_mask]
        normalized = (middle_values - black_threshold) / (white_threshold - black_threshold)
        enhanced = 0.5 * (1 + np.tanh(4 * (normalized - 0.5)))
        filtered_array[middle_mask] = black_threshold + enhanced * (white_threshold - black_threshold)
    
    # 确保所有值都在0-255范围内
    filtered_array = np.clip(filtered_array, 0, 255)
    filtered_array = filtered_array.astype(np.uint8)
    
    print(f"滤波完成，像素值范围: {filtered_array.min()} - {filtered_array.max()}")
    
    return filtered_array

def load_trained_model(model_path="mnist/mnist_mlp_model.joblib"):
    """
    加载训练好的MLP模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        MLPClassifier: 加载的模型
    """
    print(f"\n=== 加载训练好的模型 ===")
    print(f"模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        return None
    
    try:
        model = load(model_path)
        print(f"模型加载成功！")
        print(f"模型类型: {type(model).__name__}")
        print(f"隐藏层结构: {model.hidden_layer_sizes}")
        print(f"输入特征数: {model.n_features_in_}")
        print(f"输出类别数: {model.n_outputs_}")
        return model
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None

def preprocess_for_prediction(image_array):
    """
    为预测准备图像数据：归一化和展平
    
    Args:
        image_array: 28x28的图像数组
    
    Returns:
        numpy.ndarray: 预处理后的特征向量
    """
    if image_array is None:
        return None
    
    print(f"\n=== 预测数据预处理 ===")
    print(f"原始图像形状: {image_array.shape}")
    print(f"原始像素值范围: {image_array.min()} - {image_array.max()}")
    
    # 归一化：将像素值从0-255缩放到0-1
    normalized = image_array.astype('float32') / 255.0
    print(f"归一化后像素值范围: {normalized.min():.3f} - {normalized.max():.3f}")
    
    # 展平：将28x28的图像转换为784维的向量
    flattened = normalized.reshape(1, -1)  # reshape为(1, 784)用于单个样本预测
    print(f"展平后特征向量形状: {flattened.shape}")
    
    return flattened

def predict_digit(model, feature_vector):
    """
    使用训练好的模型预测数字
    
    Args:
        model: 训练好的MLP模型
        feature_vector: 预处理后的特征向量
    
    Returns:
        tuple: (预测结果, 预测概率)
    """
    if model is None or feature_vector is None:
        return None, None
    
    print(f"\n=== 模型推理 ===")
    
    try:
        # 预测类别
        prediction = model.predict(feature_vector)
        predicted_digit = prediction[0]
        
        # 预测概率
        probabilities = model.predict_proba(feature_vector)
        confidence = probabilities[0]
        
        print(f"预测结果: {predicted_digit}")
        print(f"预测置信度: {confidence[predicted_digit]:.4f} ({confidence[predicted_digit]*100:.2f}%)")
        
        # 显示所有类别的概率
        print(f"\n各数字的预测概率:")
        for digit in range(10):
            print(f"  数字 {digit}: {confidence[digit]:.4f} ({confidence[digit]*100:.2f}%)")
        
        return predicted_digit, confidence
        
    except Exception as e:
        print(f"预测时发生错误: {e}")
        return None, None

def visualize_prediction_result(original_image, processed_image, filtered_image, predicted_digit, confidence):
    """
    可视化预测结果
    
    Args:
        original_image: 原始图像
        processed_image: 处理后的28x28图像
        filtered_image: 滤波后的图像
        predicted_digit: 预测的数字
        confidence: 预测置信度数组
    """
    print(f"\n=== 预测结果可视化 ===")
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'手写数字识别结果: {predicted_digit} (置信度: {confidence[predicted_digit]*100:.2f}%)', fontsize=16)
    
    # 显示原始图像
    if original_image is not None:
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
    else:
        axes[0, 0].text(0.5, 0.5, '原始图像\n不可用', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].axis('off')
    
    # 显示处理后的28x28图像
    axes[0, 1].imshow(processed_image, cmap='gray')
    axes[0, 1].set_title('28x28 处理后图像')
    axes[0, 1].axis('off')
    
    # 显示滤波后的图像
    if filtered_image is not None:
        axes[0, 2].imshow(filtered_image, cmap='gray')
        axes[0, 2].set_title('对比度增强后图像')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].imshow(processed_image, cmap='gray')
        axes[0, 2].set_title('28x28 处理后图像')
        axes[0, 2].axis('off')
    
    # 显示放大的预测图像
    enlarged = cv2.resize(filtered_image if filtered_image is not None else processed_image, 
                         (280, 280), interpolation=cv2.INTER_NEAREST)
    axes[1, 0].imshow(enlarged, cmap='gray')
    axes[1, 0].set_title(f'放大显示\n预测结果: {predicted_digit}')
    axes[1, 0].axis('off')
    
    # 显示概率分布柱状图
    axes[1, 1].bar(range(10), confidence, alpha=0.7, color='skyblue')
    axes[1, 1].bar(predicted_digit, confidence[predicted_digit], alpha=0.9, color='red')
    axes[1, 1].set_title('各数字预测概率')
    axes[1, 1].set_xlabel('数字')
    axes[1, 1].set_ylabel('概率')
    axes[1, 1].set_xticks(range(10))
    axes[1, 1].grid(True, alpha=0.3)
    
    # 显示前3个最高概率
    top3_indices = np.argsort(confidence)[-3:][::-1]
    top3_text = "前3个最可能的数字:\n"
    for i, idx in enumerate(top3_indices):
        top3_text += f"{i+1}. 数字 {idx}: {confidence[idx]*100:.2f}%\n"
    
    axes[1, 2].text(0.1, 0.9, top3_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='top')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数 - 完整的手写数字识别推理流程
    """
    print("=== 手写数字识别推理程序 ===")
    print("本程序将加载图像，预处理后使用训练好的模型进行数字识别")
    print()
    
    # 1. 设置图像路径
    image_path = "captured_images/roi_20250729_144125.bmp"
    alternative_path = "mnist/roi_20250729_144125.bmp"
    
    # 检查图像文件
    if os.path.exists(image_path):
        target_path = image_path
    elif os.path.exists(alternative_path):
        target_path = alternative_path
    else:
        print("错误：找不到图像文件！")
        print(f"已检查路径:")
        print(f"  - {image_path}")
        print(f"  - {alternative_path}")
        return
    
    # 2. 加载和预处理图像
    processed_image = load_and_process_image(target_path, (28, 28))
    if processed_image is None:
        print("图像预处理失败！")
        return
    
    # 3. 应用对比度增强滤波（可选）
    filtered_image = enhance_contrast_filter(processed_image, black_threshold=40, white_threshold=100)
    
    # 选择使用滤波后的图像还是原始处理图像进行预测
    prediction_image = filtered_image if filtered_image is not None else processed_image
    
    # 4. 加载训练好的模型
    model = load_trained_model("mnist/mnist_mlp_model.joblib")
    if model is None:
        print("模型加载失败！")
        return
    
    # 5. 预处理图像数据用于预测
    feature_vector = preprocess_for_prediction(prediction_image)
    if feature_vector is None:
        print("预测数据预处理失败！")
        return
    
    # 6. 进行预测
    predicted_digit, confidence = predict_digit(model, feature_vector)
    if predicted_digit is None:
        print("预测失败！")
        return
    
    # 7. 显示结果
    print(f"\n=== 最终预测结果 ===")
    print(f"识别的数字: {predicted_digit}")
    print(f"置信度: {confidence[predicted_digit]*100:.2f}%")
    
    # 8. 可视化结果
    try:
        original_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        visualize_prediction_result(original_image, processed_image, filtered_image, 
                                  predicted_digit, confidence)
    except Exception as e:
        print(f"可视化时发生错误: {e}")
    
    print("\n推理完成！")
    print("提示: 关闭matplotlib窗口以结束程序")

if __name__ == "__main__":
    main()