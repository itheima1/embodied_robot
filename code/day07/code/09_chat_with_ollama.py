import ollama
response = ollama.chat(model='qwen2:latest',
messages=[{'role': 'user', 'content': '从前有座山，山里有个庙，续写一下', },])
print(response.message.content)
print(response['message']['content'])