# 语音对话程序 - 结合录音转文本和AI聊天功能
import pyaudio
import wave
import whisper
import os
import re
import random
import time
import asyncio
import edge_tts
import pygame
# pip install edge_tts pygame

# --- 录音配置 ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "voice_chat_output.wav"
TEMP_TTS_FILENAME = "ai_response.mp3"
VOICE = "zh-CN-XiaoxiaoNeural"  # AI回复使用的语音

class SimpleKernel:
    def __init__(self):
        self.categories = {}
        self.last_response = None
        self.variables = {}

    def learn(self, aiml_content):
        # 使用正则表达式匹配 AIML 格式
        pattern = re.compile(r'<category>.*?<pattern>(.*?)</pattern>(?:\s*<that>(.*?)</that>)?\s*<template>(.*?)</template>.*?</category>', re.DOTALL)
        
        matches = pattern.findall(aiml_content)
        for match in matches:
            pattern_text = match[0].strip().upper()
            that_text = match[1].strip().upper() if match[1] else None
            template_text = match[2].strip()

            # 存储模式（pattern_text）和模板（template_text）
            self.categories[(pattern_text, that_text)] = template_text

    def respond(self, input_text):
        input_text = input_text.strip().upper()

        for (pattern_text, that_text), template_text in self.categories.items():
            # 检查是否匹配模式
            if re.match(pattern_text.replace('*', '.*'), input_text):
                # 检查 that_text 是否匹配上次的响应
                if that_text is None or (self.last_response and re.match(that_text.replace('*', '.*'), self.last_response)):
                    self.last_response = input_text
                    response = self.process_template(template_text)
                    return response

        return "抱歉，我不太理解你说的话。你可以换个说法试试。"

    def process_template(self, template):
        # 处理 <random> 标签
        if '<random>' in template:
            choices = re.findall(r'<li>(.*?)</li>', template, re.DOTALL)
            template = random.choice(choices)

        # 处理 <get name="..."/> 标签
        template = re.sub(r'<get name="(.*?)"/>', lambda match: self.variables.get(match.group(1), ''), template)

        return template

    def set_variable(self, name, value):
        self.variables[name] = value

def record_audio():
    """录制音频"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    print("\n" + "="*50)
    print("  提示：按下回车键开始录音，按下 Ctrl+C 结束录音。")
    print("="*50)
    input("  请按回车键开始说话...")
    
    print("\n[*] 正在录音... 请说话。")

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("[*] 录音结束。")

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
    print("[*] 正在进行语音识别，请稍候...")
    result = model.transcribe(audio_path, language="Chinese")
    return result["text"]

def typewriter_effect(text, delay=0.05):
    """打字机效果显示文本"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # 换行

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
    print("[*] 正在生成语音回复...")
    audio_file = await text_to_speech(text)
    print("[*] 正在播放语音回复...")
    play_audio(audio_file)
    
    # 清理临时文件
    if os.path.exists(audio_file):
        os.remove(audio_file)

async def main():
    """主程序"""
    print("[*] 正在加载 Whisper 模型 (确保同级文件夹有base.pt文件)...")
    try:
        # 加载Whisper模型
        model = whisper.load_model("./base.pt") 
        print("[*] Whisper 模型加载成功！")
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        print("[!] 请检查模型文件是否存在。")
        return

    # 初始化AI聊天机器人
    aiml_content = """
<?xml version="1.0" encoding="UTF-8"?>
<aiml version="1.0.1">
    <category>
        <pattern>你好</pattern>
        <template>你好！很高兴和你聊天！</template>
    </category>
    <category>
        <pattern>HELLO</pattern>
        <template>Hi there! 很高兴见到你！</template>
    </category>
    <category>
        <pattern>你是谁</pattern>
        <template>我是你的AI助手，可以和你进行语音对话。</template>
    </category>
    <category>
        <pattern>*你好*</pattern>
        <template>你好！有什么可以帮助你的吗？</template>
    </category>
    <category>
        <pattern>*再见*</pattern>
        <template>再见！期待下次和你聊天！</template>
    </category>
    <category>
        <pattern>*拜拜*</pattern>
        <template>拜拜！祝你有美好的一天！</template>
    </category>
    <category>
        <pattern>*天气*</pattern>
        <template>我无法查看实时天气，但希望今天是个好天气！</template>
    </category>
    <category>
        <pattern>*你多大*</pattern>
        <template>
            <random>
                <li>我是AI，没有年龄概念，但我很年轻哦！</li>
                <li>年龄对AI来说是个秘密呢。</li>
                <li>我刚刚诞生不久，还在学习中。</li>
            </random>
        </template>
    </category>
    <category>
        <pattern>*喜欢*</pattern>
        <template>我喜欢和人类聊天，学习新知识！</template>
    </category>
    <category>
        <pattern>*谢谢*</pattern>
        <template>不客气！很高兴能帮到你。</template>
    </category>
    <category>
        <pattern>*帮助*</pattern>
        <template>我可以和你聊天，回答一些简单的问题。你想聊什么呢？</template>
    </category>
    <category>
        <pattern>*退出*</pattern>
        <template>好的，再见！</template>
    </category>
</aiml>
"""

    alice = SimpleKernel()
    alice.learn(aiml_content)
    
    print("\n" + "="*60)
    print("           欢迎使用语音对话系统！")
    print("="*60)
    print("说明：")
    print("1. 按回车开始录音，按Ctrl+C结束录音")
    print("2. 系统会识别你的语音并用语音回复")
    print("3. 说'退出'或'拜拜'可以结束对话")
    print("4. 请确保已安装: pip install edge_tts pygame")
    print("="*60)
    
    conversation_count = 0
    
    try:
        while True:
            conversation_count += 1
            print(f"\n--- 第 {conversation_count} 轮对话 ---")
            
            # 录音
            audio_file = record_audio()
            
            # 语音识别
            try:
                transcribed_text = transcribe_audio_with_whisper(audio_file, model)
                print(f"\n[识别结果] 你说: {transcribed_text}")
                
                # 检查是否要退出
                if any(word in transcribed_text.lower() for word in ['退出', '拜拜', '再见', 'bye']):
                    farewell_msg = "好的，再见！很高兴和你聊天！"
                    print(f"\n[AI回复] {farewell_msg}")
                    await speak_response(farewell_msg)
                    break
                
                # AI回复
                response = alice.respond(transcribed_text)
                print(f"\n[AI回复] {response}")
                await speak_response(response)
                
            except Exception as e:
                print(f"[!] 语音识别出错: {e}")
                continue
            
            finally:
                # 删除临时音频文件
                if os.path.exists(TEMP_WAVE_FILENAME):
                    os.remove(TEMP_WAVE_FILENAME)
    
    except KeyboardInterrupt:
        print("\n\n[*] 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n[!] 程序运行出错: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(TEMP_WAVE_FILENAME):
            os.remove(TEMP_WAVE_FILENAME)
        print("\n感谢使用语音对话系统！")

if __name__ == "__main__":
    asyncio.run(main())