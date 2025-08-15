# DeepSeek + MCP 集成示例

这个项目展示了如何将 DeepSeek API 与 MCP (Model Context Protocol) 服务器集成，让 DeepSeek 能够调用本地工具。

## 文件说明

- `deepseek_api_demo.py` - DeepSeek API 基础使用示例
- `mcp_client.py` - MCP 客户端基础使用示例
- `mcp_server.py` - MCP 服务器实现（提供加法工具）
- `deepseek_mcp_integration.py` - **主要集成代码**
- `simple_example.py` - 简化的使用示例

## 安装依赖

```bash
pip install openai mcp
```

## 使用步骤

### 1. 启动 MCP 服务器

首先在一个终端中启动 MCP 服务器：

```bash
python mcp_server.py
```

服务器将在 `http://localhost:8000/sse` 运行。

### 2. 运行集成示例

在另一个终端中运行示例：

```bash
# 运行完整示例
python deepseek_mcp_integration.py

# 或运行简化示例
python simple_example.py
```

## 工作原理

1. **MCP 服务器**提供工具（如加法计算）
2. **DeepSeek API**接收用户请求
3. **集成代码**将 MCP 工具转换为 DeepSeek 可理解的函数格式
4. DeepSeek 决定是否需要调用工具
5. 如果需要，集成代码调用 MCP 工具并将结果返回给 DeepSeek
6. DeepSeek 基于工具结果生成最终回复

## 示例对话

```
用户: 请帮我计算 15 + 27 的结果
助手: 我来帮您计算 15 + 27。

[调用加法工具: add(15, 27)]

计算结果是 42。
```

## 自定义扩展

### 添加新工具到 MCP 服务器

在 `mcp_server.py` 中添加新的工具函数：

```python
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b

@mcp.tool()
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    # 实现天气查询逻辑
    return f"{city}的天气是晴天"
```

### 修改 DeepSeek 配置

在 `DeepSeekMCPIntegration` 类中可以修改：
- API 密钥
- 模型名称
- 系统提示词
- MCP 服务器地址

## 注意事项

1. 确保 MCP 服务器在运行集成代码之前已启动
2. 检查 DeepSeek API 密钥是否有效
3. 网络连接需要能够访问 DeepSeek API 和本地 MCP 服务器
4. 工具调用可能会产生额外的 API 费用

## 故障排除

- **连接 MCP 服务器失败**：检查服务器是否运行在正确端口
- **DeepSeek API 错误**：验证 API 密钥和网络连接
- **工具调用失败**：检查工具参数格式是否正确