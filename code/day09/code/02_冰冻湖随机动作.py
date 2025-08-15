import gymnasium as gym
import time

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
observation, info = env.reset()

# 循环 10 步
for i in range(10):
    # 渲染当前状态
    env.render()
    
    # 从动作空间中随机选择一个动作
    # 0: 左, 1: 下, 2: 右, 3: 上
    action = env.action_space.sample()
    
    # 执行这个动作，并获取返回信息
    # observation: 新的状态（位置）
    # reward: 执行动作后获得的奖励 (到达终点为 1, 其他为 0)
    # terminated: 是否到达终点或掉入冰洞 (游戏是否结束)
    # truncated: 是否因为其他原因（如超时）结束
    # info: 额外信息
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"第 {i+1} 步: 执行动作 {action}, 到达新位置 {observation}, 获得奖励 {reward}")
    
    # 稍微暂停一下，方便观察
    time.sleep(0.5)
    
    # 如果游戏结束了（掉洞里或到终点），就跳出循环
    if terminated or truncated:
        print("游戏结束！")
        break

# 渲染最后的状态
env.render()
time.sleep(2)
env.close()