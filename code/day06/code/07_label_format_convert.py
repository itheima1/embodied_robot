import json
import os

def json_to_yolo(json_data, class_names):
    """
    将JSON格式的标注数据转换为YOLO格式
    
    Args:
        json_data: JSON格式的标注数据
        class_names: 类别名称列表
    
    Returns:
        YOLO格式的标注字符串
    """
    yolo_lines = []
    
    # 获取图像尺寸
    img_width = json_data['size']['width']
    img_height = json_data['size']['height']
    
    # 处理每个对象
    for obj in json_data['outputs']['object']:
        class_name = obj['name']
        
        # 获取类别索引
        if class_name in class_names:
            class_id = class_names.index(class_name)
        else:
            print(f"警告: 未知类别 '{class_name}'，跳过该对象")
            continue
        
        # 获取边界框坐标
        xmin = obj['bndbox']['xmin']
        ymin = obj['bndbox']['ymin']
        xmax = obj['bndbox']['xmax']
        ymax = obj['bndbox']['ymax']
        
        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # 格式化为YOLO格式字符串
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return '\n'.join(yolo_lines)

def convert_single_json_to_yolo(json_file_path, output_dir, class_names):
    """
    转换单个JSON文件为YOLO格式
    
    Args:
        json_file_path: JSON文件路径
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 转换为YOLO格式
        yolo_content = json_to_yolo(json_data, class_names)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_file_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 写入YOLO格式文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(yolo_content)
        
        print(f"转换完成: {json_file_path} -> {output_file_path}")
        
    except Exception as e:
        print(f"转换失败 {json_file_path}: {str(e)}")

def convert_directory_json_to_yolo(json_dir, output_dir, class_names):
    """
    批量转换目录中的所有JSON文件为YOLO格式
    
    Args:
        json_dir: JSON文件目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    if not os.path.exists(json_dir):
        print(f"错误: 输入目录不存在 {json_dir}")
        return
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"在目录 {json_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件，开始转换...")
    
    for json_file in json_files:
        json_file_path = os.path.join(json_dir, json_file)
        convert_single_json_to_yolo(json_file_path, output_dir, class_names)
    
    print("批量转换完成！")

def main():
    """
    主函数 - 演示如何使用转换功能
    """
    # 示例JSON数据
    sample_json_data = {
        "path": "E:\\jszn_bj\\day06\\mask_data\\000000000091.jpg",
        "outputs": {
            "object": [
                {
                    "name": "mask",
                    "bndbox": {
                        "xmin": 154,
                        "ymin": 40,
                        "xmax": 343,
                        "ymax": 268
                    }
                }
            ]
        },
        "time_labeled": 1753777251548,
        "labeled": True,
        "size": {
            "width": 500,
            "height": 333,
            "depth": 3
        }
    }
    
    # 定义类别名称（根据实际情况修改）
    class_names = ['mask','no_mask']  # 可以添加更多类别，如 ['mask', 'no_mask', 'person']
    
    print("=== JSON到YOLO格式转换工具 ===")
    print("\n示例转换:")
    
    # 转换示例数据
    yolo_result = json_to_yolo(sample_json_data, class_names)
    print(f"原始JSON数据转换结果:")
    print(yolo_result)
    
    print("\n=== 使用说明 ===")
    print("1. 单个文件转换:")
    print("   convert_single_json_to_yolo('input.json', 'output_dir', class_names)")
    print("\n2. 批量转换目录:")
    print("   convert_directory_json_to_yolo('json_dir', 'output_dir', class_names)")
    
    # 实际使用示例（取消注释以使用）
    json_input_dir = "E:\\jszn_bj\\day06\\mask_data\\outputs"  # JSON文件目录
    yolo_output_dir = "E:\\jszn_bj\\day06\\mask_data\\yolo"  # YOLO标签输出目录
    convert_directory_json_to_yolo(json_input_dir, yolo_output_dir, class_names)

if __name__ == "__main__":
    main()