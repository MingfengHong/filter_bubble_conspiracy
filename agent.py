"""
用户代理 (UserAgent)
代表具有确认偏误的信息消费者
"""
import numpy as np
from mesa import Agent


class UserAgent(Agent):
    """
    代表一个具有确认偏误的用户
    
    属性:
        belief: 当前信念，范围 [-1.0, 1.0]
        P_strength: 确认偏误强度 (0-1)，控制选择性接触的强度
        learning_rate: 信念更新速率
        history: 消费内容的历史记录
    """
    
    def __init__(self, model, initial_belief, P_strength, learning_rate=0.05):
        super().__init__(model)
        self.belief = initial_belief
        self.P_strength = P_strength
        self.learning_rate = learning_rate
        self.history = []  # 记录消费过的内容
        
    def step(self):
        """
        代理的行为步骤：
        1. 从平台获取个性化信息流
        2. 基于确认偏误选择内容
        3. 更新信念（强化螺旋）
        """
        # 1. 获取平台推荐的信息流
        feed = self.model.generate_feed(self, feed_size=10)
        
        # 2. 基于确认偏误选择内容（实现 P 机制）
        selected_item = self.select_content(feed)
        
        if selected_item is not None:
            # 3. 消费内容并更新信念
            self.update_belief(selected_item)
            self.history.append({
                'step': self.model.steps,
                'slant': selected_item['slant'],
                'belief_after': self.belief
            })
    
    def select_content(self, feed):
        """
        基于确认偏误从信息流中选择内容
        使用指数衰减函数：吸引力 = exp(-距离 * P_scaled)
        
        Args:
            feed: 平台提供的信息流列表
            
        Returns:
            选中的内容项
        """
        if not feed:
            return None
        
        # 1. 计算信息流中每个内容与用户信念的距离
        distances = np.array([abs(self.belief - item['slant']) for item in feed])
        
        # 2. 将距离转化为吸引力（使用指数衰减）
        # P_strength 从 0-1 缩放到 0-5，控制衰减速度（降低以避免过度锁定）
        P_scaled = self.P_strength * 5
        attractiveness = np.exp(-distances * P_scaled)
        
        # 3. 归一化为概率分布
        total_attractiveness = np.sum(attractiveness)
        if total_attractiveness > 0 and not np.isinf(total_attractiveness):
            probabilities = attractiveness / total_attractiveness
        else:
            # 容错：如果总和为零或溢出，使用均匀分布
            probabilities = np.ones(len(feed)) / len(feed)
        
        # 4. 根据概率选择内容
        selected_index = np.random.choice(len(feed), p=probabilities)
        return feed[selected_index]
    
    def update_belief(self, item):
        """
        基于消费的内容更新信念
        使用线性更新规则（类似 Deffuant 模型）
        
        Args:
            item: 消费的内容项
        """
        # 信念向内容倾向靠拢
        self.belief += self.learning_rate * (item['slant'] - self.belief)
        
        # 确保信念保持在 [-1, 1] 范围内
        self.belief = np.clip(self.belief, -1.0, 1.0)

