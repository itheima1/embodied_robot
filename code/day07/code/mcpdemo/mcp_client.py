from mcp import ClientSession
from mcp.client.sse import sse_client


async def run():
    # 使用SSE协议连接到服务器
    async with sse_client(url="http://localhost:8000/sse") as streams:
        # 创建客户端会话
        async with ClientSession(*streams) as session:

            # 初始化会话
            await session.initialize()

            # 列出所有可用工具
            tools = await session.list_tools()
            print("可用工具:", tools)

            # 调用加法工具
            result = await session.call_tool("add", arguments={"a": 4, "b": 5})
            print("加法结果:", result.content[0].text)


if __name__ == "__main__":
    import asyncio

    # 运行异步主函数
    asyncio.run(run())