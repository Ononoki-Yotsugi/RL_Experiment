"""
21点环境
=======
state: [玩家手牌列表], 庄家明牌
action: {叫牌(0)，停牌(1)}
reward: 胜利 1，失败 -1，平局 0
"""
import numpy as np


class BasePolicy:
    """策略基类
    """
    def act(self, obs):
        raise NotImplementedError('Policy.act Not Implemented')


class DealerPolicy(BasePolicy):
    """庄家策略

    手牌小于17要牌，否则停止
    """
    def act(self, obs):
        if obs < 17:
            return 0
        else:
            return 1


class BlackJack:
    def __init__(self):
        super().__init__()

        # 动作空间
        # 0: 要牌
        # 1: 停止
        self.action_space = (0, 1)

        # 游戏状态:
        # 0 玩家抽卡阶段，玩家停止抽卡时进入下一阶段
        # 1 庄家抽卡阶段
        # 2 结算阶段
        self.state = 0

        # 玩家与庄家的卡
        self.player_trajectory = []
        self.dealer_trajectory = []

        # 庄家策略
        self.dealer_policy = DealerPolicy()

    def reset(self):
        # 给每人发两张卡
        self.player_trajectory = []
        self.player_trajectory.append(self._get_card())
        self.player_trajectory.append(self._get_card())
        self.dealer_trajectory = []
        self.dealer_trajectory.append(self._get_card())
        self.dealer_trajectory.append(self._get_card())

        self.state = 0

        return self._get_obs()

    def step(self, action):
        if action == 0:  # 玩家抽卡
            assert self.state == 0, '只能在 0 状态抽卡'

            # 抽卡
            self.player_trajectory.append(self._get_card())

            # 检测是否爆牌
            if self._is_blast(self.player_trajectory):
                return self._get_obs(), -1, True, {}

            return self._get_obs(), 0, False, {}
        
        elif action == 1: # 玩家停止要牌
            # 进入庄家决策阶段
            self.state += 1

            # 获取庄家观测
            dealer_obs = max_point(self.dealer_trajectory)
            # 庄家决策
            action = self.dealer_policy.act(dealer_obs)
            while action == 0:
                # 庄家抽排
                self.dealer_trajectory.append(self._get_card())
                # 计算当前点数之和
                dealer_obs = max_point(self.dealer_trajectory)
                # 爆牌检测，如果庄家爆牌，玩家得到1的回报
                if self._is_blast(self.dealer_trajectory):
                    return self._get_obs(), 1, True, {}
                # 如果没有爆牌，根据当前点数计算新的动作
                action = self.dealer_policy.act(dealer_obs)

            # 庄家停止要牌，开始结算
            self.state += 1
            player_point = max_point(self.player_trajectory)
            dealer_point = max_point(self.dealer_trajectory)
            # 比较点数，计算回报
            if player_point > dealer_point:
                reward = 1
            elif player_point == dealer_point:
                reward = 0
            else:
                reward = -1
            return self._get_obs(), reward, True, {}

        else:
            raise ValueError('非法动作')

    def _get_card(self):
        """抽卡

        Returns
        -------
        int
            抽到的卡(1-10)
        """
        card = np.random.randint(1, 14)
        card = min(card, 10)
        return card

    def _get_obs(self):
        """获取观测

        Returns
        -------
        Cards : list of int
            手牌列表
        Dealer's card 1 : int
            庄家的第一张牌
        """
        return (self.player_trajectory.copy(), self.dealer_trajectory[0])

    def _is_blast(self, traj):
        """检测是否爆牌

        Parameters
        ----------
        traj : list of int
            牌列表

        Returns
        -------
        blast : bool
            如果爆牌返回 True，否则 False
        """
        return max_point(traj) > 21


def max_point(traj):
    """工具函数，计算一个牌列表的最大点数

    Parameters
    ----------
    traj : list of int
        牌列表
    
    Returns
    -------
    point : int
        最大点数
    """
    s = 0
    num_ace = 0
    for card in traj:
        if card == 1:
            num_ace += 1
            s += 11
        else:
            s += card
    
    while s > 21 and num_ace > 0:
        s -= 10
        num_ace -= 1
    
    return s

