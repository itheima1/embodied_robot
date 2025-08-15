# 结合DeepSeek API和MCP客户端的集成代码
# 需要安装依赖: pip install openai mcp

import asyncio
import json
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client


class DeepSeekMCPIntegration:
    def __init__(self, api_key: str, mcp_server_url: str = "http://localhost:8000/sse"):
        """初始化DeepSeek和MCP客户端"""
        self.deepseek_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.mcp_server_url = mcp_server_url
        self.available_tools = []
    
    async def get_mcp_tools(self):
        """获取MCP服务器可用的工具"""
        async with sse_client(url=self.mcp_server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.available_tools = tools.tools
                return self.available_tools
    
    async def call_mcp_tool(self, tool_name: str, arguments: dict):
        """调用MCP服务器的工具"""
        async with sse_client(url=self.mcp_server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result.content[0].text if result.content else None
    
    def format_tools_for_deepseek(self):
        """将MCP工具格式化为DeepSeek可理解的格式"""
        formatted_tools = []
        for tool in self.available_tools:
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"调用{tool.name}工具",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # 如果工具有输入模式，添加参数信息
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                formatted_tool["function"]["parameters"] = tool.inputSchema
            
            formatted_tools.append(formatted_tool)
        
        return formatted_tools
    
    async def chat_with_tools(self, user_message: str, system_message: str = "你是一个有用的助手，可以使用工具来帮助用户。"):
        """与DeepSeek聊天并支持工具调用"""
        # 获取可用工具
        await self.get_mcp_tools()
        tools = self.format_tools_for_deepseek()
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # 调用DeepSeek API
        response = self.deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            stream=False
        )
        
        assistant_message = response.choices[0].message
        
        # 如果DeepSeek想要调用工具
        if assistant_message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in assistant_message.tool_calls]
            })
            
            # 执行工具调用
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_result = await self.call_mcp_tool(tool_name, arguments)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"工具调用失败: {str(e)}"
                    })
            
            # 再次调用DeepSeek获取最终回复
            final_response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            
            return final_response.choices[0].message.content
        else:
            return assistant_message.content


async def main():
    """主函数示例"""
    # 初始化集成客户端
    integration = DeepSeekMCPIntegration(
        api_key="sk-6bc37dec91f84ed19278eb9c2ed9cd40"
    )
    
    print("=== DeepSeek + MCP 集成示例 ===")
    
    # 获取可用工具
    print("\n1. 获取MCP服务器可用工具:")
    try:
        tools = await integration.get_mcp_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"连接MCP服务器失败: {e}")
        print("请确保MCP服务器正在运行 (python mcp_server.py)")
        return
    
    # 测试对话
    print("\n2. 测试DeepSeek调用MCP工具:")
    test_messages = [
        "请帮我计算 15 + 27 的结果",
        "能帮我算一下 100 加 200 等于多少吗？",
        "你好，请介绍一下你自己"
    ]
    
    for message in test_messages:
        print(f"\n用户: {message}")
        try:
            response = await integration.chat_with_tools(message)
            print(f"助手: {response}")
        except Exception as e:
            print(f"处理消息时出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())