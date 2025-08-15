from mcp.server.fastmcp import FastMCP
#pip install fastmcp

# 创建MCP服务器实例
mcp = FastMCP()

#### 工具函数 ####
# 添加加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加"""
    print(f"计算 {a} 加 {b}")
    return a + b

# 腰部关节旋转控制工具
@mcp.tool()
def rotate_waist(angle: float) -> str:
    """控制机械臂腰部关节旋转
    
    Args:
        angle: 旋转角度，范围-90度到90度
    
    Returns:
        str: 执行结果信息
    """
    # 检查角度范围
    if angle < -90 or angle > 90:
        return f"错误：角度 {angle} 超出范围，腰部关节旋转范围为-90度到90度"
    
    print(f"控制腰部关节旋转到 {angle} 度")
    # 这里可以添加实际的机械臂控制代码
    return f"腰部关节已旋转到 {angle} 度"

# 夹爪控制工具
@mcp.tool()
def control_gripper(angle: float) -> str:
    """控制机械臂夹爪开合
    
    Args:
        angle: 夹爪角度，范围0度到-90度（0度为完全张开，-90度为完全闭合）
    
    Returns:
        str: 执行结果信息
    """
    # 检查角度范围
    if angle > 0 or angle < -90:
        return f"错误：角度 {angle} 超出范围，夹爪控制范围为0度到-90度"
    
    print(f"控制夹爪到 {angle} 度位置")
    # 这里可以添加实际的夹爪控制代码
    if angle == 0:
        status = "完全张开"
    elif angle == -90:
        status = "完全闭合"
    else:
        status = f"部分闭合（{abs(angle)}度）"
    
    return f"夹爪已调整到 {angle} 度位置（{status}）"

# 可以在此处添加更多工具
if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='sse')