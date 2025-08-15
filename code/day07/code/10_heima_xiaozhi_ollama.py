# 黑马小智 - 基于Ollama本地大模型的语音对话系统
import pyaudio
import wave
import whisper
import os
import time
import asyncio
import edge_tts
import pygame
import ollama
# pip install edge_tts pygame ollama

# --- 录音配置 ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "xiaozhi_input.wav"
TEMP_TTS_FILENAME = "xiaozhi_response.mp3"
VOICE = "zh-CN-XiaoxiaoNeural"  # AI回复使用的语音

# Ollama模型配置
OLLAMA_MODEL = "qwen2:latest"  # 可以根据需要修改模型名称

class HeimaXiaozhi:
    def __init__(self):
        self.conversation_history = [
            {"role": "system", "content": "你是黑马小智，一个由传智教育开发的AI助手。你性格活泼友好，善于解答编程、学习和生活方面的问题。请用简洁、亲切的语言回复，每次回复控制在100字以内。"}
        ]
    
    def chat(self, user_input):
        """与Ollama本地大模型进行对话"""
        try:
            # 添加用户输入到对话历史
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 保持对话历史在合理长度内（最多保留最近10轮对话）
            if len(self.conversation_history) > 21:  # system + 10轮对话(每轮2条消息)
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            # 调用Ollama API
            response = ollama.chat(
                model='qwen2:latest',
                messages=self.conversation_history
            )
            
            ai_response = response['message']['content']
            
            # 添加AI回复到对话历史
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            print(f"[!] Ollama API调用出错: {e}")
            return "抱歉，我现在有点忙，请稍后再试。可能需要检查Ollama服务是否正常运行。"

def record_audio():
    """录制音频"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    print("\n" + "="*50)
    print("  🎤 按下回车键开始录音，按下 Ctrl+C 结束录音")
    print("="*50)
    input("  请按回车键开始说话...")
    
    print("\n[*] 🔴 正在录音... 请说话")

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("[*] ⏹️ 录音结束")

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

def transcribe_audio_with_whisper(audio_path, model):
    """使用Whisper进行语音识别"""
    print("[*] 🧠 正在进行语音识别，请稍候...")
    result = model.transcribe(audio_path, language="Chinese")
    return result["text"]

async def text_to_speech(text, output_file=TEMP_TTS_FILENAME):
    """将文本转换为语音并保存"""
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_file)
    return output_file

def play_audio(audio_file):
    """播放音频文件"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
    except Exception as e:
        print(f"[!] 音频播放出错: {e}")

async def speak_response(text):
    """AI语音回复"""
    print("[*] 🎵 正在生成语音回复...")
    audio_file = await text_to_speech(text)
    print("[*] 🔊 正在播放语音回复...")
    play_audio(audio_file)
    
    # 清理临时文件
    if os.path.exists(audio_file):
        os.remove(audio_file)

def check_ollama_connection():
    """检查Ollama服务连接"""
    try:
        # 简单测试连接
        response = ollama.chat(
            model='qwen2:latest',
            messages=[{'role': 'user', 'content': 'hello'}]
        )
        print(f"[*] ✅ Ollama服务连接成功！模型 qwen2:latest 已就绪")
        return True
    except Exception as e:
        print(f"[!] ❌ Ollama服务连接失败: {e}")
        print("[!] 请确保Ollama服务正在运行，可以尝试运行: ollama serve")
        print("[!] 请确保已安装qwen2:latest模型，可以运行: ollama pull qwen2:latest")
        return False

async def main():
    """主程序"""
    print("[*] 🤖 正在加载 Whisper 模型 (确保同级文件夹有base.pt文件)...")
    try:
        # 加载Whisper模型
        model = whisper.load_model("./base.pt") 
        print("[*] ✅ Whisper 模型加载成功！")
    except Exception as e:
        print(f"[!] ❌ 模型加载失败: {e}")
        print("[!] 请检查模型文件是否存在。")
        return

    # 检查Ollama连接
    print("[*] 🔗 正在检查Ollama服务连接...")
    if not check_ollama_connection():
        print("[!] ❌ 无法连接到Ollama服务，程序退出")
        return

    # 初始化黑马小智
    xiaozhi = HeimaXiaozhi()
    
    print("\n" + "="*60)
    print("           🎓 欢迎使用黑马小智语音对话系统！(Ollama版)")
    print("="*60)
    print("说明：")
    print("1. 🎤 按回车开始录音，按Ctrl+C结束录音")
    print("2. 🧠 系统会识别你的语音并用本地AI智能回复")
    print("3. 🚪 说'退出'、'拜拜'或'再见'可以结束对话")
    print("4. 📦 请确保已安装: pip install edge_tts pygame ollama")
    print("5. 🔧 请确保Ollama服务正在运行: ollama serve")
    print("6. 🤖 当前使用模型: qwen2:latest")
    print("7. 🌟 由传智教育黑马程序员提供技术支持")
    print("="*60)
    
    conversation_count = 0
    
    try:
        while True:
            conversation_count += 1
            print(f"\n--- 🗣️ 第 {conversation_count} 轮对话 ---")
            
            # 录音
            audio_file = record_audio()
            
            # 语音识别
            try:
                transcribed_text = transcribe_audio_with_whisper(audio_file, model)
                print(f"\n[识别结果] 👤 你说: {transcribed_text}")
                
                # 检查是否要退出
                if any(word in transcribed_text.lower() for word in ['退出', '拜拜', '再见', 'bye', '结束']):
                    farewell_msg = "好的，再见！很高兴和你聊天，期待下次见面！记得多来黑马程序员学习哦！"
                    print(f"\n[小智回复] 🤖 {farewell_msg}")
                    await speak_response(farewell_msg)
                    break
                
                # AI回复
                print("[*] 🤔 小智正在思考...")
                response = xiaozhi.chat(transcribed_text)
                print(f"\n[小智回复] 🤖 {response}")
                await speak_response(response)
                
            except Exception as e:
                print(f"[!] ❌ 语音识别出错: {e}")
                continue
            
            finally:
                # 删除临时音频文件
                if os.path.exists(TEMP_WAVE_FILENAME):
                    os.remove(TEMP_WAVE_FILENAME)
    
    except KeyboardInterrupt:
        print("\n\n[*] 👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n[!] ❌ 程序运行出错: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(TEMP_WAVE_FILENAME):
            os.remove(TEMP_WAVE_FILENAME)
        if os.path.exists(TEMP_TTS_FILENAME):
            os.remove(TEMP_TTS_FILENAME)
        print("\n🎓 感谢使用黑马小智语音对话系统！(Ollama版)")
        print("💪 传智教育，让每一个人都有人生出彩的机会！")

if __name__ == "__main__":
    asyncio.run(main())