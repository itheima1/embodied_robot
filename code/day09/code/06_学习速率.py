# 在多大程度上相信这次的新经验”就是 学习率 (Alpha, α)


#假设你第一次去某个城市旅游，但体验非常糟糕。你到达的那天，恰好遇上恶劣天气，航班延误了数小时。接着，你预订的酒店弄错了你##的订单，让你在前台等了很久。最后，你出门想吃点东西，却发现因为是公共假期，大部分餐厅都关门了。
#你很可能会想：“这个城市真是糟透了，我再也不想来了。” 这就相当于把对这个城市的评分定为0分。
#但是，更理性的想法是：“我是不是只是运气不好，刚好赶上了糟糕的一天？” #你可能会认识到，任何城市都可能遇到恶劣天气，酒店偶尔也会出错，公共假期也是事先可以查到的。
#因此，你不会彻底否定这个城市。你对它的期待值可能会从一个中立的“5分”下降到“2分”或“3分”，并决定如果下次有机会，可能会在一#个不同的季节、做更充分的准备后再来一次，看看会不会有不一样的体验。你更新了你的看法，但并没有让一次糟糕的经历完全定义你对##它的永久印象。

#Alpha (α): 一个介于 0 和 1 之间的数字。
#α = 0：超级固执，完全不学习新知识。
#α = 1：极度健忘，新经验会完全覆盖掉旧的记忆。

#新分数 = 旧分数 + 学习率 * ( (即时奖励 + γ*未来最好分数) - 旧分数 )

import gymnasium as gym
import numpy as np
import random
import time

# 1. 创建环境
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array")

# 2. 获取环境信息
state_size = env.observation_space.n
action_size = env.action_space.n

# 3. 创建空白的“备忘录” (Q表)
q_table = np.zeros((state_size, action_size))

# 4. 定义学习中需要用到的所有参数
total_episodes = 15000        # 总共玩多少轮游戏
learning_rate = 0.1           # 学习率 (α)：我们有多相信新的经验
gamma = 0.99                  # 折扣因子 (γ)：我们有多看重未来的奖励
epsilon = 1.0                 # 初始探索率：一开始有多喜欢“瞎逛”
max_epsilon = 1.0             # 探索率上限
min_epsilon = 0.01            # 探索率下限 (即使学得很好，也偶尔要探索一下)
decay_rate = 0.001            # 探索率每次衰减多少

# --- 5. 开始训练循环 ---
for episode in range(total_episodes):
    # 每开始新一轮，都回到起点
    state, info = env.reset()
    done = False
    
    while not done:
        # 决定是“利用”还是“探索”
        if random.uniform(0, 1) > epsilon:
            # 利用：查备忘录，选分数最高的动作
            action = np.argmax(q_table[state, :])
        else:
            # 探索：随机选一个动作
            action = env.action_space.sample()

        # 执行选择的动作，并从环境中得到反馈
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- 6. 使用完整的 Q-Learning 公式更新备忘录 ---
        # 旧分数 = q_table[state, action]
        # 未来最好分数 = np.max(q_table[new_state, :])
        q_table[state, action] = q_table[state, action] + learning_rate * \
                                 (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        
        # 前往下一个位置
        state = new_state

    # 游戏结束后，降低“探索欲”，因为我们又多学到了一点知识
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("--- 训练完成！ ---")
print("最终学到的“备忘录” (Q 表):")
print(q_table)

# --- 7. 让机器人用学到的知识来表演！ ---
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
for episode in range(3):
    state, info = env.reset()
    done = False
    print(f"--- 表演赛 第 {episode + 1} 轮 ---")
    time.sleep(1)
    env.render()
    
    while not done:
        # 在表演时，我们不再探索，总是选择最优策略
        action = np.argmax(q_table[state, :])
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 渲染每一步
        time.sleep(0.3)
        env.render()

        if done:
            if reward == 1:
                print("太棒了！我找到了终点！")
            else:
                print("哎呀，掉进洞里了...")
        
        state = new_state
        
    time.sleep(2)
env.close()