# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-31893719ae5544738215cfee93c4ab7a", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "给我讲一个笑话，100字以内"},
    ],
    stream=False
)

print(response.choices[0].message.content)