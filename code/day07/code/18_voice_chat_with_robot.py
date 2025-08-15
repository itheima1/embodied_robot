# 语音控制机械臂对话系统
# 结合录音转文本和DeepSeek API及MCP客户端的集成代码
# 需要安装依赖: pip install openai mcp pyaudio wave openai-whisper

import asyncio
import json
import pyaudio
import wave
import whisper
import os
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client


# --- 录音配置 ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "voice_input.wav"


class VoiceRecorder:
    """语音录制类"""
    
    def __init__(self):
        self.whisper_model = None
        self.load_whisper_model()
    
    def load_whisper_model(self):
        """加载Whisper模型"""
        print("🎤 正在加载 Whisper 语音识别模型...")
        try:
            self.whisper_model = whisper.load_model("./base.pt")
            print("✅ Whisper 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请确保同级文件夹有base.pt文件")
            raise e
    
    def record_audio(self):
        """录制音频"""
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        
        print("\n" + "="*50)
        print("  🎤 按下回车键开始录音，按下 Ctrl+C 结束录音")
        print("="*50)
        input("  请按回车键开始录音...")
        
        print("\n🔴 正在录音... 请说话")

        try:
            while True:
                data = stream.read(CHUNK)
                frames.append(data)
        except KeyboardInterrupt:
            print("🟢 录音结束")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(TEMP_WAVE_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return TEMP_WAVE_FILENAME
    
    def transcribe_audio(self, audio_path):
        """语音转文本"""
        print("🔄 正在进行语音识别，请稍候...")
        result = self.whisper_model.transcribe(audio_path, language="Chinese")
        return result["text"].strip()
    
    def get_voice_input(self):
        """获取语音输入并转换为文本"""
        try:
            audio_file = self.record_audio()
            transcribed_text = self.transcribe_audio(audio_file)
            
            print("\n" + "="*50)
            print("  📝 语音识别结果:")
            print("="*50)
            print(f"  {transcribed_text}")
            print("\n")
            
            # 删除临时文件
            if os.path.exists(TEMP_WAVE_FILENAME):
                os.remove(TEMP_WAVE_FILENAME)
            
            return transcribed_text
        except Exception as e:
            print(f"❌ 语音识别出错: {e}")
            return None


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


async def voice_interactive_chat():
    """语音交互式聊天主函数"""
    # 初始化语音录制器
    try:
        voice_recorder = VoiceRecorder()
    except Exception as e:
        print(f"❌ 语音录制器初始化失败: {e}")
        return
    
    # 初始化集成客户端
    integration = DeepSeekMCPIntegration(
        api_key="sk-6bc37dec91f84ed19278eb9c2ed9cd40"
    )
    
    print("🤖 === 语音控制机械臂智能对话系统 ===")
    print("💡 提示: 说 '退出' 或 '结束' 退出程序")
    print("💡 提示: 说 '清空' 清空对话历史")
    print("💡 提示: 说 '工具' 查看可用工具")
    print("💡 提示: 说 '帮助' 查看帮助信息")
    print("💡 提示: 也可以按 Ctrl+C 退出程序")
    
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
    
    print("\n🎯 机械臂已准备就绪! 请开始语音指令:")
    print("-" * 50)
    
    # 开始语音交互循环
    while True:
        try:
            # 获取语音输入
            print("\n🎤 等待语音输入...")
            user_input = voice_recorder.get_voice_input()
            
            if not user_input:
                print("❌ 语音识别失败，请重试")
                continue
            
            print(f"👤 您说: {user_input}")
            
            # 处理特殊命令
            if any(keyword in user_input.lower() for keyword in ['退出', '结束', 'quit', 'exit']):
                print("👋 再见! 语音控制机械臂对话系统已退出。")
                break
            elif any(keyword in user_input.lower() for keyword in ['清空', 'clear']):
                integration.clear_history()
                continue
            elif any(keyword in user_input.lower() for keyword in ['工具', 'tools']):
                print("\n🔧 可用工具列表:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                continue
            elif any(keyword in user_input.lower() for keyword in ['帮助', 'help']):
                print("\n📖 帮助信息:")
                print("  - 直接说出自然语言指令来控制机械臂")
                print("  - 例如: '向右转动腰部30度'")
                print("  - 例如: '张开夹爪'")
                print("  - 例如: '机械臂复位'")
                print("  - 说 '退出' 退出程序")
                print("  - 说 '清空' 清空对话历史")
                print("  - 说 '工具' 查看可用工具")
                continue
            
            # 处理用户消息
            print("\n🤖 机械臂: 正在处理您的语音指令...")
            response = await integration.chat_with_tools(user_input)
            print(f"🤖 机械臂: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到 Ctrl+C，语音控制机械臂对话系统已退出。")
            break
        except Exception as e:
            print(f"\n❌ 处理语音消息时出错: {e}")
            print("请重试或说 '退出' 退出程序。")


async def main():
    """主函数"""
    await voice_interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())