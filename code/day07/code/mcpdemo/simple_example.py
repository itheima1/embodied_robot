# 简化的DeepSeek + MCP集成使用示例

import asyncio
from deepseek_mcp_integration import DeepSeekMCPIntegration


async def simple_demo():
    """简单的演示示例"""
    # 创建集成实例
    integration = DeepSeekMCPIntegration(
        api_key="sk-6bc37dec91f84ed19278eb9c2ed9cd40"
    )
    
    print("🤖 DeepSeek + MCP 简单示例")
    print("=" * 40)
    
    try:
        # 测试数学计算
        print("\n📝 测试: 让DeepSeek使用MCP工具进行计算")
        response = await integration.chat_with_tools(
            "请帮我计算 25 + 17 的结果，并解释一下计算过程"
        )
        print(f"💬 DeepSeek回复: {response}")
        
        # 测试另一个计算
        print("\n📝 测试: 复杂一点的计算")
        response = await integration.chat_with_tools(
            "我需要计算 88 + 12，然后告诉我这个结果是否是偶数"
        )
        print(f"💬 DeepSeek回复: {response}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n💡 请确保:")
        print("   1. MCP服务器正在运行: python mcp_server.py")
        print("   2. 已安装必要依赖: pip install openai mcp")


if __name__ == "__main__":
    asyncio.run(simple_demo())