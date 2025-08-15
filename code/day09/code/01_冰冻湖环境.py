#pip install gymnasium numpy

import gymnasium as gym
import time

# 创建冰冻湖环境
# is_slippery=False 表示冰面不滑，我们的移动是确定的。 更容易理解。
# render_mode='human' 让我们能看到图形化界面。
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

# 重置环境到初始状态，这会返回机器人的初始位置（状态 0）
observation, info = env.reset()
print(observation)
print(info)

# 渲染环境，显示初始状态
env.render()

# 暂停几秒钟，方便我们观察
print("环境已创建，这是初始状态。机器人位于 S。")
time.sleep(10)

# 关闭环境
env.close()