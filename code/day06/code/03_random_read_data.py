import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def read_random_images(num_images=8):
    """
    从MNIST CSV文件中随机读取多行数据，转换为20x20图像并放大到100x100显示
    
    Args:
        num_images: 要读取的图像数量，默认为8
    """
    # 读取CSV文件
    images_file = 'mnist/mnist_20x20_images.csv'
    labels_file = 'mnist/mnist_20x20_labels.csv'
    
    try:
        # 读取图像和标签CSV文件
        df_images = pd.read_csv(images_file, header=None)
        df_labels = pd.read_csv(labels_file)
        print(f"成功读取图像文件，共有 {len(df_images)} 行数据")
        print(f"成功读取标签文件，共有 {len(df_labels)} 行数据")
        
        # 随机选择多行
        random_rows = random.sample(range(len(df_images)), num_images)
        print(f"随机选择了 {num_images} 行数据: {[row + 1 for row in random_rows]}")
        
        images = []
        
        # 处理每一行数据
        for i, row_idx in enumerate(random_rows):
            # 获取该行的像素数据
            pixel_data = df_images.iloc[row_idx].values
            
            # 获取对应的标签
            label = df_labels.iloc[row_idx]['Label']
            
            # 将一维数组转换为20x20的二维数组
            image_20x20 = pixel_data.reshape(20, 20)
            
            # 将数据类型转换为uint8
            image_20x20 = image_20x20.astype(np.uint8)
            
            # 使用OpenCV将20x20图像放大到100x100
            image_100x100 = cv2.resize(image_20x20, (100, 100), interpolation=cv2.INTER_NEAREST)
            
            images.append((image_20x20, image_100x100, label))
        
        # 使用matplotlib显示所有图像
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('MNIST 20x20 图像随机展示 (放大到100x100)', fontsize=16)
        
        for i, (original, enlarged, label) in enumerate(images):
            row = i // 4
            col = i % 4
            
            # 显示放大后的图像
            axes[row, col].imshow(enlarged, cmap='gray')
            axes[row, col].set_title(f'标签: {label}\n{enlarged.shape[0]}x{enlarged.shape[1]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"成功显示 {num_images} 个图像！")
        return images
        
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e}")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None

def main():
    """
    主函数
    """
    print("=== MNIST 20x20 图像随机读取和显示程序 ===")
    print("程序将从CSV文件中随机选择8行数据，转换为图像并同时显示")
    print()
    
    # 读取并显示8个随机图像
    images = read_random_images(8)
    
    if images is not None:
         print(f"\n图像信息:")
         for i, (original, enlarged, label) in enumerate(images):
             print(f"图像 {i+1} (标签: {label}): 原始尺寸 {original.shape}, 放大尺寸 {enlarged.shape}, 像素值范围 {original.min()}-{original.max()}")
         
         print("\n提示: 关闭matplotlib窗口以结束程序")

if __name__ == "__main__":
    main()