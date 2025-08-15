import re
import random

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

        return "Sorry, I don't understand that."

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



aiml_content = """
<?xml version="1.0" encoding="UTF-8"?>
<aiml version="1.0.1">
    <category>
        <pattern>HELLO</pattern>
        <template>Hi there!</template>
    </category>
    <category>
        <pattern>HOW ARE YOU</pattern>
        <template>I'm doing well, thank you!</template>
    </category>
    <category>
        <pattern>WHAT IS YOUR NAME</pattern>
        <template>My name is Alice.</template>
    </category>
    <category>
        <pattern>你多大了</pattern>
        <template>
            <random>
                <li>哦，我今年 <get name="age"/> 岁，如花似玉的年龄。</li>
                <li>你都 <get name="age"/> 岁了，好棒。</li>
                <li><get name="age"/> 岁？我比你年轻好多呢。</li>
                <li>哦，<get name="age"/> 岁，您学到的知识比我多得多呢。</li>
            </random>
        </template>
    </category>
    <category>
        <pattern>*睡*</pattern>
        <template>我是人工智能，不需要睡觉。不过，真希望自己也能做个美梦呢。</template>
    </category>
    <category>
        <pattern>*我漂亮*</pattern>
        <template>你真的很漂亮，比白雪公主还美丽。</template>
    </category>
</aiml>
"""

# Create kernel instance and load AIML content
alice = SimpleKernel()
alice.learn(aiml_content)
alice.set_variable("age", "25")  # Set the age variable

# List of user inputs
user_inputs = ["我漂亮吗"]

# Process each user input and print the response
for user_input in user_inputs:
    print(f"Alice, you asked... >> {user_input}")
    response = alice.respond(user_input)
    print(response)