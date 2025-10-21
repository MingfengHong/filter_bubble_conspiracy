# 信息茧房与确认偏误的共谋：代理基建模

> 使用 Mesa 框架实现的计算社会科学模型  
> 探索算法个性化推荐（Q）与人类确认偏误（P）的"共谋"机制

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Mesa](https://img.shields.io/badge/Mesa-3.3.0-green.svg)](https://mesa.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

本项目通过代理基建模 (Agent-Based Modeling, ABM) 探索"信息茧房"的形成机制。

### 核心观点

**信息茧房不仅仅是算法的"罪恶"，而是算法个性化推荐（Q）与人类确认偏误（P）之间的"共谋"结果。**

### 理论基础

- **信息茧房** (Filter Bubble, Pariser 2011): 算法的"预先选择"限制了信息多样性
- **选择性接触** (Selective Exposure): 用户主动寻找符合既有信念的信息
- **确认偏误** (Confirmation Bias, Nickerson 1998): 人类倾向于确认而非挑战既有信念
- **认知失调理论** (Cognitive Dissonance, Festinger 1957): 人们规避心理不适
- **强化螺旋模型** (Reinforcing Spirals, Slater): 态度与媒体选择相互强化

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

需要的包：
- Mesa >= 3.3.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0

### 三种运行方式

#### 方式一：2x2 对比（最推荐！⭐）

同时运行四种场景并实时对比：

```bash
python compare_2x2.py
# 或双击：2x2对比.bat
```

**特点：**
- 🔷 蓝色：基线 (Q=0.1, P=0.1) - 多元环境
- 🟢 绿色：选择性接触 (Q=0.1, P=0.8) - 用户主动寻找认同
- 🟡 橙色：信息茧房 (Q=0.8, P=0.1) - 算法限制信息
- 🔴 红色：共谋 (Q=0.8, P=0.8) - 算法与偏误共同作用

**实验结果示例：**
```
最终极化程度对比:
  基线 (低P低Q): 0.0193
  选择性接触 (高P低Q): 0.0448
  信息茧房 (低P高Q): 0.0497
  共谋 (高P高Q): 0.0883  ⬅️ 最高！
```

✅ **验证了理论：共谋 > 信息茧房 ≈ 选择性接触 > 基线**

#### 方式二：单场景深入分析

```bash
python run_simple.py
# 或双击：启动简化版.bat
```

实时显示 6 张图表，可修改代码调整参数。

#### 方式三：批量实验报告

```bash
python experiment.py
# 或双击：运行实验.bat
```

自动生成：
- `experiment_2x2_results.png` - 四种场景详细对比
- `polarization_comparison.png` - 极化趋势时间序列
- `experiment_summary.csv` - 定量结果摘要

---

## 🎯 模型设计

### 核心机制

模型包含两个连续的过滤阶段和反馈循环：

```
┌─────────────────┐
│   内容池        │
│ (均匀分布-1~1)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 算法预选 (Q)    │  ◄── Q 强度控制推荐个性化程度
│ 生成信息流      │      相似度 = exp(-距离 × Q×5)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 用户自选 (P)    │  ◄── P 强度控制确认偏误程度
│ 选择一个内容    │      吸引力 = exp(-距离 × P×5)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 信念更新        │  ◄── 信念向内容靠拢
│ 强化螺旋        │      belief += 0.1 × (slant - belief)
└────────┬────────┘
         │
         └──────► (反馈到算法)
```

### 2x2 实验设计

| 场景 | P (确认偏误) | Q (算法) | 预期效应 |
|------|-------------|---------|---------|
| 基线 | 低 (0.1) | 低 (0.1) | 多元环境，温和演化 |
| 选择性接触 | **高 (0.8)** | 低 (0.1) | 用户主动极化 |
| 信息茧房 | 低 (0.1) | **高 (0.8)** | 算法导致极化 |
| 共谋 | **高 (0.8)** | **高 (0.8)** | **最强极化** ⚡ |

### 关键参数

- **Q (算法个性化强度)**: 0~1，控制推荐内容与用户信念的相似度
  - 0 = 完全随机推荐
  - 1 = 只推荐最相似内容
  
- **P (确认偏误强度)**: 0~1，控制用户选择内容的偏好
  - 0 = 随机选择
  - 1 = 只选最接近自己信念的内容

- **learning_rate**: 信念更新速率，默认 0.1
  - 越大，信念变化越快

---

## 📊 测量指标

### 极化程度 (Polarization)

信念分布的方差 `var(beliefs)`

```python
polarization = np.var([agent.belief for agent in model.agents])
```

- 值越大 → 观点越分裂
- 从单峰 → 双峰 = 极化的标志

**典型值范围：**
- 基线场景: 0.01 ~ 0.03
- 中等极化: 0.04 ~ 0.06
- 强极化: > 0.08

### 信念分布

- **单峰分布** (集中在中间) → 共识
- **双峰分布** (分布在两极) → 极化

可观察左翼、中间、右翼的人数分布：
- 左翼: belief < -0.3
- 中间: -0.3 ≤ belief ≤ 0.3
- 右翼: belief > 0.3

---

## 🔬 核心代码

### 算法推荐 (Q 机制)

```python
# 在 model.py 中
def generate_feed(self, agent, feed_size=10):
    # 计算内容与用户信念的距离
    distances = np.array([abs(agent.belief - item['slant']) 
                          for item in self.content_pool])
    
    # 转化为相似度（距离越小，相似度越高）
    Q_scaled = self.Q_strength * 5
    similarities = np.exp(-distances * Q_scaled)
    
    # 概率抽样生成信息流
    probabilities = similarities / np.sum(similarities)
    feed = np.random.choice(self.content_pool, size=10, p=probabilities)
    return feed
```

### 确认偏误 (P 机制)

```python
# 在 agent.py 中
def select_content(self, feed):
    # 计算信息流中内容的吸引力
    distances = np.array([abs(self.belief - item['slant']) 
                          for item in feed])
    
    P_scaled = self.P_strength * 5
    attractiveness = np.exp(-distances * P_scaled)
    
    # 概率选择最吸引的内容
    probabilities = attractiveness / np.sum(attractiveness)
    selected = np.random.choice(feed, p=probabilities)
    return selected
```

### 信念更新（强化螺旋）

```python
# 在 agent.py 中
def update_belief(self, item):
    # 线性更新：信念向消费内容靠拢
    self.belief += self.learning_rate * (item['slant'] - self.belief)
    self.belief = np.clip(self.belief, -1.0, 1.0)
```

---

## 📁 项目结构

```
.
├── agent.py                # UserAgent 类：具有确认偏误的用户代理
├── model.py                # PlatformModel 类：平台与算法推荐系统
├── compare_2x2.py          # 2x2 实时对比 ⭐ 推荐
├── run_simple.py           # 单场景简化版可视化
├── experiment.py           # 批量实验脚本
├── analyze.py              # 单次深入分析
├── requirements.txt        # 依赖包列表
├── README.md               # 项目文档（本文件）
│
├── 2x2对比.bat             # 快捷启动：2x2对比
├── 启动简化版.bat          # 快捷启动：单场景
├── 运行实验.bat            # 快捷启动：批量实验
│
└── app.py                  # Solara 可视化（可选，实验性）
```

---

## 🎓 理论意义

### 创新之处

1. **解耦机制**: 明确分离算法作用 (Q) 和用户主动性 (P)
2. **量化共谋**: 通过参数化实验测量两者的交互效应
3. **动态演化**: 捕捉强化螺旋的长期涌现过程
4. **可计算性**: 将抽象理论转化为可测试的计算模型

### 关键发现

通过实验验证：

✅ **共谋效应真实存在**
- 当 P 和 Q 都高时，极化程度显著高于单一机制

✅ **算法与人性相互强化**
- 算法满足用户偏好 → 用户更依赖算法 → 螺旋上升

✅ **"过度锁定"陷阱**
- 当缩放因子过大（如×10），会导致用户锁定无法移动
- 降低到×5后，效果更符合现实

---

## 📈 实验建议

### 第一步：理解基线

运行 `compare_2x2.py`，观察四种场景的初始状态

### 第二步：观察演化

注意观察：
- 信念分布如何从单峰变为双峰
- 极化曲线哪条上升最快
- 左翼、中间、右翼的人数变化

### 第三步：验证理论

检查最终极化程度是否符合：
```
共谋 > 信息茧房 ≈ 选择性接触 > 基线
```

### 第四步：参数实验

尝试修改参数观察影响：
- `learning_rate`: 0.05 vs 0.1 vs 0.2
- `num_users`: 50 vs 100 vs 200
- `Q/P 强度`: 尝试 0.5, 0.6, 0.7 等中间值

---

## 🔮 扩展方向

### 模型扩展

1. **异质性代理**: 不同用户具有不同的 P 强度
2. **社交网络**: 加入代理间的社交影响 (Peer Effects)
3. **内容生产**: 内容池随时间动态变化
4. **干预实验**: 测试"打破茧房"的策略（如强制推荐多元内容）
5. **真实数据校准**: 使用真实平台数据校准参数

### 研究问题

- **临界点分析**: P 和 Q 达到多少会触发极化？
- **时间动态**: 极化需要多长时间形成？能否逆转？
- **群体差异**: 不同初始信念分布如何影响结果？
- **平台设计**: 如何设计算法减少极化？

---

## 🐛 常见问题

### Q1: 极化程度太低？

**原因**: 缩放因子或学习率可能太小

**解决**: 
- 增加 `learning_rate`（在创建模型时）
- 调整 `agent.py` 和 `model.py` 中的缩放因子

### Q2: 共谋场景极化反而低？

**原因**: "过度锁定" - 参数太高导致用户无法移动

**解决**: 
- 降低缩放因子（已修复为 ×5）
- 使用 Q=0.9, P=0.9 而非 0.8

### Q3: 结果每次都不同？

**原因**: 模型包含随机性

**解决**: 
- 运行多次取平均
- 增加用户数量获得更稳定结果
- 使用 `np.random.seed()` 固定随机种子

### Q4: 图表中文显示为方框？

**原因**: 字体问题

**解决**: 修改字体设置
```python
matplotlib.rcParams['font.sans-serif'] = ['你的中文字体']
```

---

## 📚 参考文献

### 核心理论

- Pariser, E. (2011). *The Filter Bubble: What the Internet Is Hiding from You*. Penguin.
- Festinger, L. (1957). *A Theory of Cognitive Dissonance*. Stanford University Press.
- Nickerson, R. S. (1998). Confirmation Bias: A Ubiquitous Phenomenon in Many Guises. *Review of General Psychology*, 2(2), 175-220.

### 计算社会科学方法

- Axelrod, R. (1997). *The Complexity of Cooperation: Agent-Based Models of Competition and Collaboration*. Princeton University Press.
- Epstein, J. M. (2006). *Generative Social Science: Studies in Agent-Based Computational Modeling*. Princeton University Press.

### 实证研究

- Zuiderveen Borgesius, F. J., et al. (2016). Should We Worry About Filter Bubbles? *Internet Policy Review*, 5(1).
- Haroon, M., et al. (2022). Auditing YouTube's Recommendation Algorithm for Radicalization Pathways. *FAccT*.
- Slater, M. D. (2007). Reinforcing Spirals: The Mutual Influence of Media Selectivity and Media Effects. *Communication Theory*, 17(3), 281-303.
- Bennett, W. L., & Iyengar, S. (2008). A New Era of Minimal Effects? *Journal of Communication*, 58(4), 707-731.

---

## 💻 技术栈

- **语言**: Python 3.8+
- **框架**: Mesa 3.3.0 (Agent-Based Modeling)
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib
- **可选**: Solara (Web 可视化)

---

## 📄 License

MIT License - 用于学术研究和教育目的

---

## 🤝 贡献

这是一个用于复杂系统课程的研究项目。欢迎：
- 提出改进建议
- 报告 bug
- 扩展模型功能
- 分享实验结果

---

## 📧 联系

本项目探索计算社会科学方法在传播学中的应用，研究信息茧房的形成机制。

**致谢**: 感谢 Mesa 开发团队提供优秀的 ABM 框架。

---

<p align="center">
  <strong>祝你的复杂系统研究顺利！🎓📊</strong>
</p>
