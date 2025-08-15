import gymnasium as gym
import numpy as np

#在 Q 表的第 14 行（代表位置 14），第 2 列（代表向右走）的那个 0，变成了 #1！机器人学到了它的第一条知识：“当我在位置14时，向右走是个非常好的选择！”
env = gym.make('FrozenLake-v1', is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# --- 让我们手动模拟一个成功的例子 ---

# 假设机器人当前在终点 G 的左边，也就是位置 14 (state=14)
# 终点 G 是位置 15
current_state = 14
# 它选择向右走 (action=2)
action_to_take = 2 

# 模拟执行这个动作
# 在真实环境中，这会是 env.step() 的结果
# 我们手动设置一下，方便理解
new_state = 15
reward = 1.0 # 到达终点，获得奖励！

# --- 开始更新我们的“备忘录” ---
print(f"更新前，位置 {current_state} 的分数: {q_table[current_state, :]}")

# 应用我们最简单的学习规则：Q(state, action) = reward
q_table[current_state, action_to_take] = reward

print(f"更新后，位置 {current_state} 的分数: {q_table[current_state, :]}")
print("\n完整的 Q 表:")
print(q_table)

env.close()