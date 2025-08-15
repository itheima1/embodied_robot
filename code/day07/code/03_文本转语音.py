import asyncio
import edge_tts
# pip install edge_tts

# 要转换的文字
TEXT = "你好，欢迎使用参加传智教育的具身智能培训！这是一个非常简单的例子。"
# 输出的音频文件名
OUTPUT_FILE = "test_audio.mp3"
# 选择一个中文发音人（例如：zh-CN-XiaoxiaoNeural 是一个女声）
VOICE = "zh-CN-shaanxi-XiaoniNeural"

#zh-CN-XiaoxiaoNeural (女声, 推荐)
#zh-CN-XiaoyiNeural (女声)
#zh-CN-YunjianNeural (男声, 推荐)
#zh-CN-YunxiNeural (男声)
#zh-CN-YunxiaNeural (男声, 情感)
#zh-CN-shaanxi-XiaoniNeural (女声, 陕西话方言)

async def main():
    """主要函数，用于生成语音"""
    print("正在生成语音...")
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)
    print(f"语音已保存到文件：{OUTPUT_FILE}")

if __name__ == "__main__":

    # 运行异步函数
    asyncio.run(main())