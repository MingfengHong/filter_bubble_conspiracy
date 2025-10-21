"""
平台模型 (PlatformModel)
管理信息环境、用户代理和推荐算法
"""
import numpy as np
from mesa import Model
from mesa import DataCollector
from agent import UserAgent


class PlatformModel(Model):
    """
    信息平台模型，实现算法推荐机制
    
    属性:
        num_users: 用户数量
        Q_strength: 算法个性化强度 (0-1)
        P_strength: 确认偏误强度 (0-1)
        content_pool: 信息内容池
        learning_rate: 信念更新速率
    """
    
    def __init__(self, num_users=100, Q_strength=0.5, P_strength=0.5, 
                 learning_rate=0.05, content_pool_size=1000):
        super().__init__()
        self.num_users = num_users
        self.Q_strength = Q_strength
        self.P_strength = P_strength
        self.learning_rate = learning_rate
        
        # 1. 初始化内容池：在 [-1, 1] 范围内均匀分布
        self.content_pool = self._create_content_pool(content_pool_size)
        
        # 2. 创建用户代理
        # 初始信念设置为围绕0的正态分布（温和状态）
        for i in range(self.num_users):
            initial_belief = np.random.normal(0, 0.2)
            initial_belief = np.clip(initial_belief, -1.0, 1.0)
            
            UserAgent(
                model=self,
                initial_belief=initial_belief,
                P_strength=self.P_strength,
                learning_rate=self.learning_rate
            )
        
        # 3. 设置数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Polarization": self.calculate_polarization,
                "Mean_Belief": self.calculate_mean_belief,
                "Belief_Std": self.calculate_belief_std
            },
            agent_reporters={
                "Belief": "belief"
            }
        )
    
    def _create_content_pool(self, size):
        """
        创建多元的信息内容池
        
        Args:
            size: 内容池大小
            
        Returns:
            内容项列表，每项包含 id 和 slant（倾向性）
        """
        content_pool = []
        for i in range(size):
            content_pool.append({
                'id': i,
                'slant': np.random.uniform(-1.0, 1.0)
            })
        return content_pool
    
    def generate_feed(self, agent, feed_size=10):
        """
        为特定用户生成个性化信息流（实现算法 Q 机制）
        使用指数衰减函数：相似度 = exp(-距离 * Q_scaled)
        
        Args:
            agent: 目标用户代理
            feed_size: 信息流大小
            
        Returns:
            推荐的内容项列表
        """
        # 1. 计算内容池中所有内容与用户信念的距离
        distances = np.array([abs(agent.belief - item['slant']) 
                            for item in self.content_pool])
        
        # 2. 将距离转化为相似度（使用指数衰减）
        # Q_strength 从 0-1 缩放到 0-5（降低以避免过度锁定）
        Q_scaled = self.Q_strength * 5
        similarities = np.exp(-distances * Q_scaled)
        
        # 3. 归一化为概率分布
        total_similarity = np.sum(similarities)
        if total_similarity > 0 and not np.isinf(total_similarity):
            probabilities = similarities / total_similarity
        else:
            # 容错：使用均匀分布
            probabilities = np.ones(len(self.content_pool)) / len(self.content_pool)
        
        # 4. 根据概率抽样生成信息流（不重复）
        feed_size = min(feed_size, len(self.content_pool))
        indices = np.random.choice(
            len(self.content_pool), 
            size=feed_size, 
            replace=False, 
            p=probabilities
        )
        feed = [self.content_pool[i] for i in indices]
        
        return feed
    
    def calculate_polarization(self):
        """
        计算整体极化程度
        使用信念的方差作为极化指标
        
        Returns:
            极化程度（方差）
        """
        beliefs = [agent.belief for agent in self.agents]
        return np.var(beliefs)
    
    def calculate_mean_belief(self):
        """计算平均信念"""
        beliefs = [agent.belief for agent in self.agents]
        return np.mean(beliefs)
    
    def calculate_belief_std(self):
        """计算信念标准差"""
        beliefs = [agent.belief for agent in self.agents]
        return np.std(beliefs)
    
    def step(self):
        """
        模型的一个时间步
        """
        self.datacollector.collect(self)
        # Mesa 3.x: 让所有代理执行一步
        self.agents.shuffle_do("step")

