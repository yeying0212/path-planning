

'''训练程序'''
import pylab as pl
from copy import deepcopy
from env import PathPlanning, AlgorithmAdaptation
import pandas as pd
from sac import SAC








'''模式设置''' 
MAX_EPISODE = 1000        # 总的训练/评估次数
render = False           # 是否可视化


'''环境算法设置'''
env = PathPlanning(max_search_steps=300)
env = AlgorithmAdaptation(env)
agent = SAC(env.observation_space, env.action_space, memory_size=800000) # 实例化强化学习算法


    
'''强化学习训练/测试仿真'''
steps_out =[]
mean_reward_out = []

for episode in range(MAX_EPISODE):
    ## 重置回合奖励
    ep_reward = 0
    
    ## 获取初始观测
    obs = env.reset()
    
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        if render:
            env.render()
        
        # 决策
        act = agent.select_action(obs)  # 随机策略

        # 仿真
        next_obs, reward, done, info,agent1_pos = env.step(act)
        ep_reward += reward
        
        # 缓存
        agent.store_memory((obs, act, reward, next_obs, done))
        
        # 优化
        agent.learn()
        
        # 回合结束
        if info["done"]:
            mean_reward = ep_reward / (steps + 1)
            print('回合: ', episode,'| 累积奖励: ', round(ep_reward, 2),'| 平均奖励: ', round(mean_reward, 2),'| 状态: ', info,'| 步数: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    #end for
    steps_out.append(steps)
    mean_reward_out.append(mean_reward)
    
#end for
df1 = pd.DataFrame(steps_out)
df2 = pd.DataFrame(mean_reward_out)
df1.to_excel('steps_out.xlsx', index=False)
df2.to_excel('mean_reward_out.xlsx', index=False)
agent.save("Model.pkl")




