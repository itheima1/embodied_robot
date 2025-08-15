# 与机械臂进行持续交互对话的程序
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
        self.conversation_history = []  # 保存对话历史
    
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
    
    async def chat_with_tools(self, user_message: str, system_message: str = "你是一个智能机械臂助手，可以通过工具控制机械臂的各种动作。你需要理解用户的指令并调用相应的工具来控制机械臂。请用友好和专业的语气回复用户。"):
        """与DeepSeek聊天并支持工具调用"""
        # 获取可用工具
        if not self.available_tools:
            await self.get_mcp_tools()
        tools = self.format_tools_for_deepseek()
        
        # 构建消息历史
        messages = [{"role": "system", "content": system_message}]
        
        # 添加对话历史（保留最近10轮对话）
        messages.extend(self.conversation_history[-20:])  # 保留最近20条消息（约10轮对话）
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
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
                    print(f"🤖 正在执行: {tool_name}({arguments})")
                    tool_result = await self.call_mcp_tool(tool_name, arguments)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                except Exception as e:
                    print(f"❌ 工具调用失败: {e}")
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
            
            final_content = final_response.choices[0].message.content
        else:
            final_content = assistant_message.content
        
        # 更新对话历史
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": final_content})
        
        return final_content
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("💭 对话历史已清空")


async def interactive_chat():
    """交互式聊天主函数"""
    # 初始化集成客户端
    integration = DeepSeekMCPIntegration(
        api_key="sk-6bc37dec91f84ed19278eb9c2ed9cd40"
    )
    
    print("🤖 === 机械臂智能对话系统 ===")
    print("💡 提示: 输入 'quit' 或 'exit' 退出程序")
    print("💡 提示: 输入 'clear' 清空对话历史")
    print("💡 提示: 输入 'tools' 查看可用工具")
    print("💡 提示: 输入 'help' 查看帮助信息")
    
    # 获取可用工具
    print("\n🔧 正在连接MCP服务器...")
    try:
        tools = await integration.get_mcp_tools()
        print(f"✅ 成功连接! 可用工具数量: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"❌ 连接MCP服务器失败: {e}")
        print("请确保MCP服务器正在运行 (python 16_real_robot_mcp_server.py)")
        return
    
    print("\n🎯 机械臂已准备就绪! 请输入您的指令:")
    print("-" * 50)
    
    # 开始交互循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 您: ").strip()
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见! 机械臂对话系统已退出。")
                break
            elif user_input.lower() in ['clear', '清空', 'c']:
                integration.clear_history()
                continue
            elif user_input.lower() in ['tools', '工具', 't']:
                print("\n🔧 可用工具列表:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                continue
            elif user_input.lower() in ['help', '帮助', 'h']:
                print("\n📖 帮助信息:")
                print("  - 直接输入自然语言指令来控制机械臂")
                print("  - 例如: '向右转动腰部30度'")
                print("  - 例如: '张开夹爪'")
                print("  - 例如: '机械臂复位'")
                print("  - 输入 'quit' 退出程序")
                print("  - 输入 'clear' 清空对话历史")
                print("  - 输入 'tools' 查看可用工具")
                continue
            elif not user_input:
                continue
            
            # 处理用户消息
            print("\n🤖 机械臂: 正在处理您的指令...")
            response = await integration.chat_with_tools(user_input)
            print(f"🤖 机械臂: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到 Ctrl+C，机械臂对话系统已退出。")
            break
        except Exception as e:
            print(f"\n❌ 处理消息时出错: {e}")
            print("请重试或输入 'quit' 退出程序。")


async def main():
    """主函数"""
    await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())