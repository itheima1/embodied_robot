import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 同样，我们用 tensorflow.keras.datasets 来方便地加载数据
# pip install tensorflow 
import tensorflow as tf
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import time

def load_mnist_data():
    """
    加载MNIST数据集
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    print("正在加载 MNIST 数据集...")
    data = np.load("mnist.npz")
    # 提取数据
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print("数据集加载完成！")
    
    # 打印数据集基本信息
    print(f"训练集图像形状: {x_train.shape}")
    print(f"训练集标签形状: {y_train.shape}")
    print(f"测试集图像形状: {x_test.shape}")
    print(f"测试集标签形状: {y_test.shape}")
    print(f"像素值范围: {x_train.min()} - {x_train.max()}")
    print(f"标签类别: {np.unique(y_train)}")
    
    return x_train, y_train, x_test, y_test

def visualize_random_samples(x_data, y_data, num_samples=16, title="MNIST 数据样本"):
    """
    随机选择并可视化MNIST数据样本
    
    Args:
        x_data: 图像数据
        y_data: 标签数据
        num_samples: 要显示的样本数量
        title: 图表标题
    """
    # 随机选择样本索引
    random_indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    # 计算子图布局
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    # 创建图表
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    # 如果只有一行或一列，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 显示每个样本
    for i, idx in enumerate(random_indices):
        row = i // cols
        col = i % cols
        
        # 显示图像
        axes[row, col].imshow(x_data[idx], cmap='gray')
        axes[row, col].set_title(f'标签: {y_data[idx]}\n索引: {idx}')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 打印选中样本的详细信息
    print(f"\n随机选择的 {num_samples} 个样本信息:")
    for i, idx in enumerate(random_indices):
        print(f"样本 {i+1}: 索引={idx}, 标签={y_data[idx]}, 图像形状={x_data[idx].shape}")

def analyze_label_distribution(y_train, y_test):
    """
    分析并可视化标签分布
    
    Args:
        y_train: 训练集标签
        y_test: 测试集标签
    """
    print("\n=== 数据集标签分布分析 ===")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    print("训练集标签分布:")
    for digit, count in zip(train_unique, train_counts):
        print(f"  数字 {digit}: {count} 个样本")
    
    print("\n测试集标签分布:")
    for digit, count in zip(test_unique, test_counts):
        print(f"  数字 {digit}: {count} 个样本")
    
    # 创建标签分布柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(train_unique, train_counts, alpha=0.7, color='blue')
    ax1.set_title('训练集标签分布')
    ax1.set_xlabel('数字类别')
    ax1.set_ylabel('样本数量')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(test_unique, test_counts, alpha=0.7, color='red')
    ax2.set_title('测试集标签分布')
    ax2.set_xlabel('数字类别')
    ax2.set_ylabel('样本数量')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def display_single_image(image, label, title="MNIST 图像"):
    """
    显示单个28x28的MNIST图像
    
    Args:
        image: 28x28的图像数组
        label: 图像对应的标签
        title: 图像标题
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f'{title}\n标签: {label}\n图像形状: {image.shape}')
    plt.axis('off')
    plt.show()
    print(f"显示图像 - 标签: {label}, 形状: {image.shape}, 像素值范围: {image.min()}-{image.max()}")

def preprocess_data(x_train, x_test):
    """
    数据预处理：归一化和展平
    
    Args:
        x_train: 训练集图像数据
        x_test: 测试集图像数据
    
    Returns:
        tuple: (x_train_processed, x_test_processed)
    """
    print("\n=== 数据预处理 ===")
    print(f"原始训练数据形状: {x_train.shape}")
    print(f"原始测试数据形状: {x_test.shape}")
    
    # 归一化：将像素值从0-255缩放到0-1
    x_train_normalized = x_train.astype('float32') / 255.0
    x_test_normalized = x_test.astype('float32') / 255.0
    
    # 展平：将28x28的图像转换为784维的向量
    x_train_flattened = x_train_normalized.reshape(x_train_normalized.shape[0], -1)
    x_test_flattened = x_test_normalized.reshape(x_test_normalized.shape[0], -1)
    
    print(f"归一化后像素值范围: {x_train_normalized.min():.3f} - {x_train_normalized.max():.3f}")
    print(f"展平后训练数据形状: {x_train_flattened.shape}")
    print(f"展平后测试数据形状: {x_test_flattened.shape}")
    
    return x_train_flattened, x_test_flattened

def train_mlp_classifier(x_train, y_train, x_test, y_test):
    """
    使用MLPClassifier训练神经网络模型
    
    Args:
        x_train: 训练集特征
        y_train: 训练集标签
        x_test: 测试集特征
        y_test: 测试集标签
    
    Returns:
        MLPClassifier: 训练好的模型
    """
    print("\n=== 神经网络模型训练 ===")
    
    # 创建MLPClassifier模型
    # hidden_layer_sizes: 隐藏层结构，这里使用两个隐藏层，分别有100和50个神经元
    # max_iter: 最大迭代次数
    # random_state: 随机种子，确保结果可重现
    # verbose: 显示训练过程
    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 10),
        max_iter=300,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print("模型参数:")
    print(f"  隐藏层结构: {mlp.hidden_layer_sizes}")
    print(f"  最大迭代次数: {mlp.max_iter}")
    print(f"  激活函数: {mlp.activation}")
    print(f"  求解器: {mlp.solver}")
    
    # 训练模型
    print("\n开始训练模型...")
    start_time = time.time()
    mlp.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n训练完成！训练时间: {training_time:.2f} 秒")
    print(f"实际迭代次数: {mlp.n_iter_}")
    
    # 预测
    print("\n进行预测...")
    y_train_pred = mlp.predict(x_train)
    y_test_pred = mlp.predict(x_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n=== 模型性能评估 ===")
    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 详细分类报告
    print("\n=== 测试集详细分类报告 ===")
    print(classification_report(y_test, y_test_pred))
    
    return mlp, y_test_pred

def visualize_predictions(x_test_original, y_test, y_pred, num_samples=12):
    """
    可视化预测结果
    
    Args:
        x_test_original: 原始测试图像数据
        y_test: 真实标签
        y_pred: 预测标签
        num_samples: 显示的样本数量
    """
    print("\n=== 预测结果可视化 ===")
    
    # 随机选择样本
    random_indices = np.random.choice(len(x_test_original), num_samples, replace=False)
    
    # 计算子图布局
    rows = 3
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    fig.suptitle('神经网络预测结果展示', fontsize=16)
    
    for i, idx in enumerate(random_indices):
        row = i // cols
        col = i % cols
        
        # 显示图像
        axes[row, col].imshow(x_test_original[idx], cmap='gray')
        
        # 设置标题，显示真实标签和预测标签
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        color = 'green' if true_label == pred_label else 'red'
        
        axes[row, col].set_title(
            f'真实: {true_label}\n预测: {pred_label}',
            color=color,
            fontsize=10
        )
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 统计正确和错误的预测
    correct_predictions = np.sum(y_test == y_pred)
    total_predictions = len(y_test)
    
    print(f"预测统计:")
    print(f"  正确预测: {correct_predictions} / {total_predictions}")
    print(f"  错误预测: {total_predictions - correct_predictions} / {total_predictions}")
    print(f"  准确率: {correct_predictions / total_predictions:.4f}")

def main():
    """
    主函数 - 演示MNIST数据集的加载、可视化和神经网络训练
    """
    # 加载数据
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    # 显示单个图像示例
    print("\n=== 单个图像展示 ===")
    print("展示训练集中的第一个图像:")
    display_single_image(x_train[0], y_train[0], "训练集第一个图像")
    
    print("\n展示测试集中的第一个图像:")
    display_single_image(x_test[0], y_test[0], "测试集第一个图像")
    
    # 可视化训练集中的随机样本
    print("\n=== 训练集随机样本可视化 ===")
    visualize_random_samples(x_train, y_train, 16, "MNIST 训练集随机样本 (28x28像素)")
    
    # 可视化测试集中的随机样本
    print("\n=== 测试集随机样本可视化 ===")
    visualize_random_samples(x_test, y_test, 9, "MNIST 测试集随机样本 (28x28像素)")
    
    # 分析标签分布
    analyze_label_distribution(y_train, y_test)
    
    # 数据预处理
    x_train_processed, x_test_processed = preprocess_data(x_train, x_test)
    
    # 训练神经网络模型
    mlp_model, y_pred = train_mlp_classifier(x_train_processed, y_train, x_test_processed, y_test)
    
    # 可视化预测结果
    visualize_predictions(x_test, y_test, y_pred, 12)
    
    # 保存模型
    print("\n=== 保存模型 ===")
    model_filename = 'mnist_mlp_model.joblib'
    dump(mlp_model, model_filename)
    print(f"模型已保存到: {model_filename}")
    
    print("\n程序执行完成！")
    print("提示: 关闭所有matplotlib窗口以结束程序")
    
    return mlp_model

if __name__ == "__main__":
    main()

