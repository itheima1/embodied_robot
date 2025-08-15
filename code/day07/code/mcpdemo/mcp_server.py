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

# 可以在此处添加更多工具
if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='sse')