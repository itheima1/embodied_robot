# 黑马小智 - 基于Ollama本地大模型的语音对话系统 (GUI版本)
import sys
import os
import time
import asyncio
import threading
from datetime import datetime

import pyaudio
import wave
import whisper
import edge_tts
import pygame
import ollama

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QWidget, QScrollArea,
    QFrame, QMessageBox, QProgressBar, QSplitter
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QIcon

# pip install PyQt5 edge_tts pygame ollama pyaudio whisper

# --- 录音配置 ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "xiaozhi_input.wav"
TEMP_TTS_FILENAME = "xiaozhi_response.mp3"
VOICE = "zh-CN-XiaoxiaoNeural"  # AI回复使用的语音

# Ollama模型配置
OLLAMA_MODEL = "qwen2:latest"

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
            return f"抱歉，我现在有点忙，请稍后再试。错误信息: {str(e)}"

class AudioRecorderThread(QThread):
    """录音线程"""
    finished = pyqtSignal(str)  # 录音完成信号
    error = pyqtSignal(str)     # 错误信号
    
    def __init__(self):
        super().__init__()
        self.is_recording = False
        
    def run(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                          input=True, frames_per_buffer=CHUNK)
            frames = []
            
            self.is_recording = True
            while self.is_recording:
                data = stream.read(CHUNK)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 保存录音文件
            wf = wave.open(TEMP_WAVE_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            self.finished.emit(TEMP_WAVE_FILENAME)
            
        except Exception as e:
            self.error.emit(f"录音出错: {str(e)}")
    
    def stop_recording(self):
        self.is_recording = False

class SpeechRecognitionThread(QThread):
    """语音识别线程"""
    finished = pyqtSignal(str)  # 识别完成信号
    error = pyqtSignal(str)     # 错误信号
    
    def __init__(self, audio_file, whisper_model):
        super().__init__()
        self.audio_file = audio_file
        self.whisper_model = whisper_model
        
    def run(self):
        try:
            result = self.whisper_model.transcribe(self.audio_file, language="Chinese")
            self.finished.emit(result["text"])
        except Exception as e:
            self.error.emit(f"语音识别出错: {str(e)}")

class ChatThread(QThread):
    """AI对话线程"""
    finished = pyqtSignal(str)  # 对话完成信号
    error = pyqtSignal(str)     # 错误信号
    
    def __init__(self, user_input, xiaozhi):
        super().__init__()
        self.user_input = user_input
        self.xiaozhi = xiaozhi
        
    def run(self):
        try:
            response = self.xiaozhi.chat(self.user_input)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(f"AI对话出错: {str(e)}")

class TTSThread(QThread):
    """文本转语音线程"""
    finished = pyqtSignal(str)  # TTS完成信号
    error = pyqtSignal(str)     # 错误信号
    
    def __init__(self, text):
        super().__init__()
        self.text = text
        
    def run(self):
        try:
            # 使用asyncio在线程中运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def generate_speech():
                communicate = edge_tts.Communicate(self.text, VOICE)
                await communicate.save(TEMP_TTS_FILENAME)
                return TEMP_TTS_FILENAME
            
            audio_file = loop.run_until_complete(generate_speech())
            loop.close()
            
            self.finished.emit(audio_file)
        except Exception as e:
            self.error.emit(f"语音合成出错: {str(e)}")

class HeimaXiaozhiGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.xiaozhi = HeimaXiaozhi()
        self.whisper_model = None
        self.conversation_count = 0
        
        # 线程对象
        self.recorder_thread = None
        self.recognition_thread = None
        self.chat_thread = None
        self.tts_thread = None
        
        self.init_ui()
        self.init_model()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("🎓 黑马小智语音对话系统 (GUI版)")
        self.setGeometry(100, 100, 900, 700)
        
        # 设置窗口图标和样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QTextEdit {
                border: 2px solid #e1e8ed;
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
                background-color: white;
            }
            QPushButton {
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                transform: translateY(0px);
            }
            #recordBtn {
                background-color: #4CAF50;
            }
            #recordBtn:hover {
                background-color: #45a049;
            }
            #stopBtn {
                background-color: #f44336;
            }
            #stopBtn:hover {
                background-color: #da190b;
            }
            #clearBtn {
                background-color: #2196F3;
            }
            #clearBtn:hover {
                background-color: #1976D2;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            #titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #1976D2;
                padding: 10px;
            }
            #statusLabel {
                font-size: 12px;
                color: #666;
                padding: 5px;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title_label = QLabel("🎓 黑马小智语音对话系统")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 对话显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(400)
        main_layout.addWidget(self.chat_display)
        
        # 状态显示
        self.status_label = QLabel("🤖 正在初始化系统...")
        self.status_label.setObjectName("statusLabel")
        main_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("🎤 开始录音")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.start_recording)
        button_layout.addWidget(self.record_btn)
        
        self.stop_btn = QPushButton("⏹️ 停止录音")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("🗑️ 清空对话")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_chat)
        button_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(button_layout)
        
        # 底部信息
        info_label = QLabel("💪 传智教育黑马程序员 - 让每一个人都有人生出彩的机会！")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; font-size: 12px; padding: 10px;")
        main_layout.addWidget(info_label)
        
        # 添加欢迎信息
        self.add_system_message("🎓 欢迎使用黑马小智语音对话系统！\n\n使用说明：\n1. 🎤 点击'开始录音'按钮开始说话\n2. ⏹️ 点击'停止录音'结束录音\n3. 🧠 系统会自动识别语音并智能回复\n4. 🔊 AI回复会自动播放语音\n\n🌟 由传智教育黑马程序员提供技术支持")
        
    def init_model(self):
        """初始化模型"""
        def load_models():
            try:
                # 加载Whisper模型
                self.status_label.setText("🤖 正在加载Whisper模型...")
                self.whisper_model = whisper.load_model("./base.pt")
                
                # 检查Ollama连接
                self.status_label.setText("🔗 正在检查Ollama服务连接...")
                response = ollama.chat(
                    model='qwen2:latest',
                    messages=[{'role': 'user', 'content': 'hello'}]
                )
                
                self.status_label.setText("✅ 系统初始化完成，可以开始对话了！")
                self.record_btn.setEnabled(True)
                
            except Exception as e:
                self.status_label.setText(f"❌ 初始化失败: {str(e)}")
                QMessageBox.critical(self, "初始化错误", 
                                   f"系统初始化失败：\n{str(e)}\n\n请检查：\n1. base.pt文件是否存在\n2. Ollama服务是否运行\n3. qwen2:latest模型是否安装")
        
        # 在后台线程中加载模型
        threading.Thread(target=load_models, daemon=True).start()
        
    def add_system_message(self, message):
        """添加系统消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"<div style='color: #666; font-size: 12px; margin: 10px 0;'>[{timestamp}] {message}</div>"
        self.chat_display.append(formatted_message)
        
    def add_user_message(self, message):
        """添加用户消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"<div style='background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 10px;'><b>👤 你说 [{timestamp}]:</b><br>{message}</div>"
        self.chat_display.append(formatted_message)
        
    def add_ai_message(self, message):
        """添加AI消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"<div style='background-color: #f3e5f5; padding: 10px; margin: 5px 0; border-radius: 10px;'><b>🤖 小智回复 [{timestamp}]:</b><br>{message}</div>"
        self.chat_display.append(formatted_message)
        
    def start_recording(self):
        """开始录音"""
        if not self.whisper_model:
            QMessageBox.warning(self, "警告", "系统尚未初始化完成，请稍候再试！")
            return
            
        self.conversation_count += 1
        self.status_label.setText(f"🔴 正在录音... (第{self.conversation_count}轮对话)")
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 启动录音线程
        self.recorder_thread = AudioRecorderThread()
        self.recorder_thread.finished.connect(self.on_recording_finished)
        self.recorder_thread.error.connect(self.on_recording_error)
        self.recorder_thread.start()
        
    def stop_recording(self):
        """停止录音"""
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.status_label.setText("⏹️ 正在停止录音...")
            
    def on_recording_finished(self, audio_file):
        """录音完成处理"""
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("🧠 正在进行语音识别...")
        
        # 启动语音识别线程
        self.recognition_thread = SpeechRecognitionThread(audio_file, self.whisper_model)
        self.recognition_thread.finished.connect(self.on_recognition_finished)
        self.recognition_thread.error.connect(self.on_recognition_error)
        self.recognition_thread.start()
        
    def on_recording_error(self, error_msg):
        """录音错误处理"""
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("❌ 录音失败")
        QMessageBox.critical(self, "录音错误", error_msg)
        
    def on_recognition_finished(self, recognized_text):
        """语音识别完成处理"""
        self.add_user_message(recognized_text)
        
        # 检查是否要退出
        if any(word in recognized_text.lower() for word in ['退出', '拜拜', '再见', 'bye', '结束']):
            farewell_msg = "好的，再见！很高兴和你聊天，期待下次见面！记得多来黑马程序员学习哦！"
            self.add_ai_message(farewell_msg)
            self.generate_and_play_speech(farewell_msg)
            return
            
        self.status_label.setText("🤔 小智正在思考...")
        
        # 启动AI对话线程
        self.chat_thread = ChatThread(recognized_text, self.xiaozhi)
        self.chat_thread.finished.connect(self.on_chat_finished)
        self.chat_thread.error.connect(self.on_chat_error)
        self.chat_thread.start()
        
    def on_recognition_error(self, error_msg):
        """语音识别错误处理"""
        self.status_label.setText("❌ 语音识别失败")
        QMessageBox.critical(self, "识别错误", error_msg)
        
    def on_chat_finished(self, ai_response):
        """AI对话完成处理"""
        self.add_ai_message(ai_response)
        self.generate_and_play_speech(ai_response)
        
    def on_chat_error(self, error_msg):
        """AI对话错误处理"""
        self.status_label.setText("❌ AI对话失败")
        self.add_ai_message(f"抱歉，出现了错误：{error_msg}")
        
    def generate_and_play_speech(self, text):
        """生成并播放语音"""
        self.status_label.setText("🎵 正在生成语音回复...")
        
        # 启动TTS线程
        self.tts_thread = TTSThread(text)
        self.tts_thread.finished.connect(self.on_tts_finished)
        self.tts_thread.error.connect(self.on_tts_error)
        self.tts_thread.start()
        
    def on_tts_finished(self, audio_file):
        """TTS完成处理"""
        self.status_label.setText("🔊 正在播放语音回复...")
        
        def play_audio():
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # 等待播放完成
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                pygame.mixer.quit()
                
                # 清理临时文件
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    
                # 更新状态
                self.status_label.setText("✅ 对话完成，可以继续录音")
                
            except Exception as e:
                self.status_label.setText(f"❌ 音频播放失败: {str(e)}")
        
        # 在后台线程播放音频
        threading.Thread(target=play_audio, daemon=True).start()
        
    def on_tts_error(self, error_msg):
        """TTS错误处理"""
        self.status_label.setText("❌ 语音合成失败")
        QMessageBox.warning(self, "语音合成错误", error_msg)
        
    def clear_chat(self):
        """清空对话"""
        reply = QMessageBox.question(self, "确认清空", "确定要清空所有对话记录吗？", 
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.chat_display.clear()
            self.conversation_count = 0
            # 重置对话历史
            self.xiaozhi = HeimaXiaozhi()
            self.add_system_message("🗑️ 对话记录已清空，可以开始新的对话了！")
            
    def closeEvent(self, event):
        """关闭事件处理"""
        # 清理临时文件
        for temp_file in [TEMP_WAVE_FILENAME, TEMP_TTS_FILENAME]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # 停止所有线程
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait()
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("黑马小智语音对话系统")
    app.setApplicationVersion("2.0")
    
    # 创建主窗口
    window = HeimaXiaozhiGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()