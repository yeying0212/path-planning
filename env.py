import numpy as np
import gym
import pylab as pl
from gym import spaces
from copy import deepcopy
from collections import deque
from scipy.integrate import odeint
from shapely import geometry as geo
from shapely.plotting import plot_polygon
from shapely.ops import nearest_points
from math import sqrt, pow, acos
import random


class Map:
    size = np.array([[-10.0, -10.0], [10.0, 10.0]])  # x, y最小值; x, y最大值
    start_pos = np.array([-7, 7])  # 起点坐标
    end_pos = np.array([8, -8])  # 终点坐标
    obstacles = [  # 障碍物, 要求为 geo.Polygon 或 带buffer的 geo.Point/geo.LineString
        geo.Polygon([(-8, 5), (-6, 5), (-6, 3), (-8, 3)]),
        geo.Polygon([(6, 0), (8, 0), (8, -2), (6, -2)]),
        geo.Polygon([(-6,-3), (-3,-3), (-3, -5), (-6, -5)]),

    ]
    def update(cls,x):
        cls.poly1 =  np.array([(-8+(3*x), 5), (-6+(3*x), 5), (-6+(3*x), 3), (-8+(3*x), 3)])
        cls.poly2 =  np.array([(6-(3*x), 0), (8-(3*x), 0), (8-(3*x), -2), (6-(3*x), -2)])

        cls.obstacles = [                 # 障碍物, 要求为 geo.Polygon 或 带buffer的 geo.Point/geo.LineString
        geo.Polygon(cls.poly1),
        geo.Polygon([(-6, -3), (-3, -3), (-3, -5), (-6, -5)]),
        geo.Polygon(cls.poly2),
    ] 

    @classmethod
    def show(cls):
        pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
        pl.mpl.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号

        pl.close('all')
        pl.figure('Map')
        pl.clf()
    
        # 障碍物
        for o in cls.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)

        # 起点终点
        pl.scatter(cls.start_pos[0], cls.start_pos[1], s=30, c='k', marker='x', label='起点')
        pl.scatter(cls.end_pos[0], cls.end_pos[1], s=30, c='k', marker='o', label='终点')
  
        pl.legend(loc='best').set_draggable(True) # 显示图例
        pl.axis('equal')                          # 平均坐标
        pl.xlabel("x")                            # x轴标签
        pl.ylabel("y")                            # y轴标签
        pl.xlim(cls.size[0][0], cls.size[1][0])   # x范围
        pl.ylim(cls.size[0][1], cls.size[1][1])   # y范围
        pl.title('Map')                           # 标题
        pl.grid()                                 # 生成网格
        pl.grid(alpha=0.3,ls=':')                 # 改变网格透明度，和网格线的样式、、
        pl.show(block=True)

        


# 静态环境
class PathPlanning(gym.Env):
    

    # 地图设置
    MAP = Map()

    def __init__(self, max_search_steps=300, use_old_gym=True):

        self.map = self.MAP
        self.max_episode_steps = max_search_steps
        np_low=np.array([-10,-10,-10,-10])
        np_high=np.array([10,10,10,10])
        self.observation_space = spaces.Box(low=np_low,high=np_high, dtype=pl.float32)
        self.action_space = spaces.Box(low=np.array([-1,-1]),high=np.array([1,1]), dtype=pl.float32)
        self.__render_flag = True
        self.__reset_flag = True
        self.__old_gym = use_old_gym




    def reset(self):
        self.__reset_flag = False
        self.time_steps = 0 
        self.traj = []
          
        #self.map.end_pos =pl.array(self.find_valid_xy())
        #self.map.start_pos = pl.array(self.find_valid_xy())

        self.agent_pos = np.array(self.map.start_pos)
        
        self.obs = np.array([10,10, 10,10])
        self.obs[0] =  self.map.end_pos[0] - self.map.start_pos[0]
        self.obs[1] =  self.map.end_pos[1] - self.map.start_pos[1]
        self.per_act = np.array([100, 100])

        """start_num = random.randint(0, 3)
        end_num   = random.randint(0, 3)
        while end_num==start_num:
            end_num = random.randint(0, 3)
        self.map.start_pos = ten_array[start_num]
        self.map.end_pos   = ten_array[end_num]"""

        # New Gym: obs, info
        # Old Gym: obs
        if self.__old_gym:
            return self.obs
        return self.obs, {}
    
    def step(self, act):
        """
        转移模型 1
        Pos_new = act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new)
        转移模型 2
        Pos_new = Pos_old + act, act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new-Pos_old)
        """
        self.now_act = act
        assert not self.__reset_flag, "调用step前必须先reset"
        # 状态转移
        obs = pl.clip(self.obs, self.observation_space.low, self.observation_space.high)
        self.agent_pos=pl.clip(self.agent_pos+act*0.3,[-10,-10], [10,10])
        agent1_pos = self.agent_pos
        obs[0] =  self.map.end_pos[0] - self.agent_pos[0]
        obs[1] =  self.map.end_pos[1] - self.agent_pos[1]
        min_dis = 100
        for o in self.map.obstacles:
            dis = geo.Point(self.agent_pos).distance(o)
            if dis < min_dis:
                min_dis = dis
                o_min = o
        p1,_ = nearest_points(o_min, geo.Point(self.agent_pos))
        p1 =p1.xy
        obs[2] = p1[0][0]-self.agent_pos[0]
        obs[3] = p1[1][0]-self.agent_pos[1]
        obs = np.array(obs)
        self.time_steps += 1
        # 计算奖励
        rew, done,info = self.get_reward(obs)
        # 回合终止
        self.obs = deepcopy(obs)
        truncated = self.time_steps >= self.max_episode_steps
        if truncated or done:
            info["done"] = True
            self.__reset_flag = True
        else:
            info["done"] = False
        # 更新状态
        if  (self.time_steps//20) % 2 == 0  :
            x = self.time_steps % 20 
        else :
            x = 20-(self.time_steps % 20)
        self.map.update(0.1*x)
        self.per_act = np.array(act)
        self.obs = deepcopy(obs)
        # New Gym: obs, rew, done, truncated, info
        # Old Gym: obs, rew, done, info
        if self.__old_gym:
            return obs, rew, done, info,agent1_pos
        return obs, rew, done, truncated, info
    
    def get_reward(self, obs):
        rew =0
        done = False
        pol_ext = geo.LinearRing(geo.Polygon([(-10, 10), (10,10), (10,-10), (-10, -10)]).exterior.coords)
        d = pol_ext.project(geo.Point(self.agent_pos))
        p = pol_ext.interpolate(d)
        closest_point_coords = list(p.coords)[0]
        #地图边界
        if  np.linalg.norm((closest_point_coords[0]-self.agent_pos[0],closest_point_coords[1]-self.agent_pos[1])) < 0.3:
            rew += -5
        #是否向前
        if np.linalg.norm((obs[0],obs[1])) +0.25< np.linalg.norm((self.obs[0],self.obs[1])):
            rew += 1
        elif np.linalg.norm((obs[0],obs[1]))-0.1> np.linalg.norm((self.obs[0],self.obs[1])):
            rew += -2
        #障碍物
        if 0.3 < np.linalg.norm((obs[2],obs[3])) < 0.8:
            rew += -5
        elif np.linalg.norm((obs[2],obs[3]))<0.3:
            rew += -1000
        elif  0.8<np.linalg.norm((obs[2],obs[3])) :
            rew += 0.5
        #转角
            if (self.per_act[0]<10):
                pi = 3.1415
                vector_prod = self.per_act[0] * self.now_act[0] + self.per_act[1] * self.now_act[1]
                length_prod = sqrt(pow(self.per_act[0], 2) + pow(self.per_act[1], 2)) * sqrt(pow(self.now_act[0], 2) + pow(self.now_act[1], 2))
                cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
                dtheta =(acos(cos) / pi) * 180
                if 90>dtheta > 45:
                    rew += -1
                elif dtheta > 90:
                    rew += -5
                elif 0< dtheta < 45:
                    rew += 0.5
                
        #终点
        if np.linalg.norm((self.map.end_pos-self.agent_pos)) < 0.5:
            rew += 5000
            done=True
        info = {}
        info['done'] =done
        
        return rew, done,info
    
    def render(self, mode="human"):
        """绘图, 必须放step前面"""
        assert not self.__reset_flag, "调用render前必须先reset"

        if self.__render_flag:
            self.__render_flag = False
            pl.ion()       # 打开交互绘图, 只能开一次

        # 清除原图像
        pl.clf() 
        # 障碍物
        for o in self.map.obstacles[0:2]:
            plot_polygon(o, facecolor='c', edgecolor='k', add_points=False)
        o = self.map.obstacles[2]
        plot_polygon(o, facecolor='c', edgecolor='k', add_points=False,label='障碍物')
        plot_polygon(geo.Point(self.agent_pos).buffer(0.8), facecolor='y',  edgecolor='y',add_points=False,label='警戒区')
        plot_polygon(geo.Point(self.agent_pos).buffer(0.3), facecolor='r', edgecolor='r', add_points=False,label='危险区')
        # 起点终点
        pl.scatter(self.map.start_pos[0], self.map.start_pos[1], s=30, c='k', marker='x', label='起点')
        pl.scatter(self.map.end_pos[0], self.map.end_pos[1], s=30, c='k', marker='o', label='终点')
        # 轨迹
        self.traj.append(self.agent_pos.tolist())
        new_lst = [item for sublist in self.traj for item in sublist]
        pl.plot(new_lst[::2], new_lst[1::2], label='path', color='b')

        pl.scatter(self.agent_pos[0],self.agent_pos[1],  s=1, c='k')
        #pl.legend(loc='best').set_draggable(True) # 显示图例
        pl.legend(loc='best')
        pl.axis('equal')                          # 平均坐标
        pl.xlabel("x")                            # x轴标签
        pl.ylabel("y")                            # y轴标签
        pl.xlim(self.map.size[0][0], self.map.size[1][0]) # x范围
        pl.ylim(self.map.size[0][1], self.map.size[1][1]) # y范围
        pl.title('Path Planning')                 # 标题
        pl.grid()                                 # 生成网格
        pl.grid(alpha=0.3,ls=':')                 # 改变网格透明度，和网格线的样式
        
        pl.pause(0.1)                           # 暂停0.01秒
        pl.ioff()                                 # 禁用交互模式

    def close(self):
        """关闭绘图"""
        self.__render_flag = True
        pl.close()



# 环境-算法适配
class AlgorithmAdaptation(gym.ActionWrapper):
    def __init__(self, env):
        super(AlgorithmAdaptation, self).__init__(env)
        assert isinstance(env.action_space, spaces.Box), '只用于Box动作空间'
  
    # 将神经网络输出转换成gym输入形式
    def action(self, action): 
        # 连续情况 scale action [-1, 1] -> [lb, ub]
        lb, ub = self.action_space.low, self.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)
        action = pl.clip(action, lb, ub)
        return action

    # 将gym输入形式转换成神经网络输出形式
    def reverse_action(self, action):
        # 连续情况 normalized action [lb, ub] -> [-1, 1]
        lb, ub = self.action_space.low, self.action_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return pl.clip(action, -1.0, 1.0)
       


if __name__ == '__main__':
    Map.show()