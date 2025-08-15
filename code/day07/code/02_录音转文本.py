# demo.py (使用 openai-whisper 库的版本)
import pyaudio
import wave
import whisper # 使用这个库，而不是 transformers.pipeline
import os
# 依赖的库有， pip install openai-whisper 
# pip install transformers accelerate torch

# --- 录音配置 ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "interactive_output.wav"

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    print("\n" + "="*50)
    print("  提示：按下回车键开始录音，按下 Ctrl+C 结束录音。")
    print("="*50)
    input("  请按回车键继续...")
    
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
    print("[*] 正在进行语音识别，请稍候...")
    result = model.transcribe(audio_path, language="Chinese")
    return result["text"]

# --- 主程序入口 ---
if __name__ == "__main__":
    print("[*] 正在加载 Whisper 模型 (确保同级文件夹有base.pt文件)...")
    try:
        # 这里使用的是 whisper.load_model()
        model = whisper.load_model("./base.pt") 
        print("[*] Whisper 模型加载成功！")
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        print("[!] 请检查模型文件是否存在。")
        exit()

    try:
        audio_file = record_audio()
        transcribed_text = transcribe_audio_with_whisper(audio_file, model)
        
        print("\n" + "="*50)
        print("  识别结果:")
        print("="*50)
        print(transcribed_text)
        print("\n")

        # 可选：删除临时文件
        os.remove(TEMP_WAVE_FILENAME)

    except Exception as e:
        print(f"[!] 程序运行出错: {e}")