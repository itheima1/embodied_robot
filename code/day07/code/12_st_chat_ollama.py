import streamlit as st
import ollama
import time

# 页面配置
st.set_page_config(
    page_title="黑马小智 - 网页版AI对话",
    page_icon="🤖",
    layout="wide"
)

# 页面标题
st.title("🤖 黑马小智 - 网页版AI对话系统")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    model_name = st.selectbox(
        "选择模型",
        ["qwen2:latest", "llama2", "mistral"],
        index=0
    )
    
    st.markdown("### 📝 使用说明")
    st.markdown("""
    1. 在下方输入框中输入您的问题
    2. 点击发送或按Ctrl+Enter发送消息
    3. AI将为您提供智能回复
    4. 支持多轮对话，保持上下文
    """)
    
    st.markdown("### 🔧 系统要求")
    st.markdown("""
    - 确保Ollama服务正在运行
    - 已安装所选模型
    - 运行命令: `ollama serve`
    """)
    
    # 清空对话按钮
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示对话历史
st.subheader("💬 对话历史")
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 用户输入
user_input = st.chat_input("请输入您的问题...")

# 处理用户输入
if user_input:
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 显示AI思考状态
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 正在思考...")
        
        try:
            # 准备对话历史
            messages_for_ollama = [
                {"role": "system", "content": "你是java之父。请用简洁、亲切的语言回复。"}
            ]
            
            # 添加历史对话（保留最近10轮）
            recent_messages = st.session_state.messages[-20:] if len(st.session_state.messages) > 20 else st.session_state.messages
            messages_for_ollama.extend(recent_messages)
            
            # 调用Ollama API
            response = ollama.chat(
                model=model_name,
                messages=messages_for_ollama
            )
            
            ai_response = response['message']['content']
            
            # 显示AI回复
            message_placeholder.markdown(ai_response)
            
            # 添加AI回复到历史
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            error_message = f"❌ 连接Ollama服务失败: {str(e)}\n\n请确保：\n1. Ollama服务正在运行 (`ollama serve`)\n2. 已安装所选模型 (`ollama pull {model_name}`)\n3. 网络连接正常"
            message_placeholder.markdown(error_message)
            
            # 添加错误信息到历史
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# 页面底部信息
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("对话轮数", len([msg for msg in st.session_state.messages if msg["role"] == "user"]))

with col2:
    st.metric("当前模型", model_name)

with col3:
    st.metric("消息总数", len(st.session_state.messages))

# 页脚
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 50px;'>
    <p>🎓 由传智教育黑马程序员提供技术支持</p>
    <p>💪 传智教育，让每一个人都有人生出彩的机会！</p>
</div>
""", unsafe_allow_html=True)

# 自动滚动到底部的JavaScript
st.markdown("""
<script>
var element = window.parent.document.querySelector('.main');
element.scrollTop = element.scrollHeight;
</script>
""", unsafe_allow_html=True)