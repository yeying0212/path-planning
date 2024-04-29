

'''测试程序'''
import torch
import pylab as pl
from copy import deepcopy
from env import PathPlanning, AlgorithmAdaptation
import pandas as pd
from sac import SAC


pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
pl.close('all')                                          # 关闭所有窗口


'''模式设置''' 
MAX_EPISODE = 1        # 总的训练/评估次数
render = True           # 是否可视化训练/评估过程(仿真速度会降几百倍)
agentpos =[]

'''环境算法设置'''
env = PathPlanning()
env = AlgorithmAdaptation(env)
agent = SAC(env.observation_space, env.action_space, memory_size=1)
agent.load("Model.pkl")


    
'''强化学习训练/测试仿真'''
for episode in range(MAX_EPISODE):
    ## 获取初始观测
    obs = env.reset()
    
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        if render:
            env.render()
        
        # 决策
        act = agent.select_action(obs)

        # 仿真
        next_obs, _, _, info,agent1_pos = env.step(act)
        
        # 回合结束
        if info["done"]:
            print('回合: ', episode,'| 状态: ', info,'| 步数: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
        agentpos.append(agent1_pos)    
    
    #end for
    df = pd.DataFrame(agentpos)
    df.to_excel('agentpos.xlsx', index=False)
#end for






