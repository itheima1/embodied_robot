import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)

# 获取有多少个位置（状态）和多少个方向（动作）
state_size = env.observation_space.n
print(state_size)
action_size = env.action_space.n
print(action_size)

print(f"冰湖上的位置总数: {state_size}")
print(f"机器人可以移动的方向数: {action_size}")

# 创建我们的“备忘录” (Q表)，并用 0 填满
# 这就像一本 16 页（每个位置一页），每页有 4 行（每个方向一行）的空白备忘录
q_table = np.zeros((state_size, action_size))

print("\n机器人的初始“备忘录” (Q 表):")
print(q_table)

env.close()