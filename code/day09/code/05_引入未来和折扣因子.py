# (位置A, 动作X)的分数 = 即时奖励 + Gamma * (从新位置出发，能得到的最好分数)
# 一个有远见的人，不仅会记录眼前的得失，还会评估一个选择的未来潜力。

#在陌生的城市里探索。
#选择A：走进一家冰淇淋店。你能立刻得到快乐（即时奖励）。
#选择B：走上一座桥。桥上什么都没有（没有即时奖励），但你隐约看到桥的对面就是你要找的著名景点！
#短视的人会选A。但有远见的人会选B，因为他知道，选择B虽然现在没好处，但未来有巨大的潜力。


# (位置A, 动作X)的分数 = 即时奖励 + Gamma * (从新位置出发，能得到的最好分数)
                                #0.9

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



# 接上一步的 Q 表
# q_table[14, 2] 已经是 1 了



# --- 我们的新参数 ---
gamma = 0.95 # 折扣因子，代表我们很有远见

# --- 模拟倒数第二步 ---
current_state = 10
action_to_take = 1 # 向下
new_state = 14
reward = 0 # 这一步没有奖励

# 从新位置(14)出发，能得到的最好分数是多少？
# 我们查一下 Q 表的第 14 行，找到最大值
max_future_q = np.max(q_table[new_state, :]) # 这会得到 1.0

# --- 应用我们带有“远见”的新公式 ---
# 新分数 = 即时奖励 + gamma * 未来最好分数
new_q_value = reward + gamma * max_future_q

print(f"计算出的新分数: 0 + {gamma} * {max_future_q} = {new_q_value}")

print(f"\n更新前，位置 {current_state} 的分数: {q_table[current_state, :]}")
q_table[current_state, action_to_take] = new_q_value
print(f"更新后，位置 {current_state} 的分数: {q_table[current_state, :]}")

# 机器人学到了新知识：“在位置10向下走也不错！虽然眼前没好处，但它能把我带到一个很有前途的地方！”
# 这个知识(0.95)会像涟漪一样，一步步从终点反向传播开来。


env.close()