"""
Mesa 可视化服务器 (基于 Solara 的新版 Mesa 3.x API)
提供实时的信念分布和极化趋势可视化
"""
import solara
from mesa.visualization import SolaraViz
from model import PlatformModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np


def belief_distribution_chart(model):
    """
    绘制信念分布直方图
    """
    beliefs = [agent.belief for agent in model.agents]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 绘制直方图
    ax.hist(beliefs, bins=20, range=(-1, 1), 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('信念值', fontsize=10)
    ax.set_ylabel('用户数量', fontsize=10)
    ax.set_title(f'信念分布 (步数: {model.steps})', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    
    # 添加统计信息
    mean_belief = np.mean(beliefs)
    std_belief = np.std(beliefs)
    polarization = np.var(beliefs)
    
    stats_text = f'均值: {mean_belief:.3f}\n标准差: {std_belief:.3f}\n极化: {polarization:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8)
    
    plt.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


def polarization_chart(model):
    """
    绘制极化程度随时间变化的图表
    """
    # 获取历史数据
    model_data = model.datacollector.get_model_vars_dataframe()
    
    if len(model_data) == 0:
        solara.Markdown("## 等待数据...")
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 绘制极化趋势
    ax.plot(model_data.index, model_data['Polarization'], 
            'r-', linewidth=2, label='极化程度')
    
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('极化程度 (方差)', fontsize=10)
    ax.set_title('极化程度随时间变化', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


def mean_belief_chart(model):
    """
    绘制平均信念随时间变化的图表
    """
    model_data = model.datacollector.get_model_vars_dataframe()
    
    if len(model_data) == 0:
        solara.Markdown("## 等待数据...")
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(model_data.index, model_data['Mean_Belief'], 
            'b-', linewidth=2, label='平均信念')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('平均信念', fontsize=10)
    ax.set_title('平均信念随时间变化', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def ModelInfo(model):
    """
    显示模型参数和场景信息
    """
    # 判断场景类型
    if model.Q_strength < 0.3 and model.P_strength < 0.3:
        scenario = "🔷 基线场景 (低P低Q) - 多元环境，开放心态"
    elif model.Q_strength < 0.3 and model.P_strength >= 0.6:
        scenario = "🟢 选择性接触 (高P低Q) - 用户主动寻找认同"
    elif model.Q_strength >= 0.6 and model.P_strength < 0.3:
        scenario = "🟡 信息茧房 (低P高Q) - 算法限制信息接触"
    elif model.Q_strength >= 0.6 and model.P_strength >= 0.6:
        scenario = "🔴 共谋场景 (高P高Q) - 算法与偏误共同作用"
    else:
        scenario = "⚪ 中等强度场景"
    
    markdown_text = f"""
## 模型参数

- **算法个性化强度 (Q):** {model.Q_strength:.2f}
- **确认偏误强度 (P):** {model.P_strength:.2f}
- **用户数量:** {model.num_users}
- **当前步数:** {model.steps}

---

### 当前场景

**{scenario}**

---

### 实验建议

1. **基线 (Q=0.1, P=0.1)**: 观察多元环境下的信念演化
2. **选择性接触 (Q=0.1, P=0.8)**: 观察用户主动寻找认同的效应
3. **信息茧房 (Q=0.8, P=0.1)**: 观察算法限制信息的效应
4. **共谋 (Q=0.8, P=0.8)**: 观察算法与偏误共同作用的最强效应
"""
    
    solara.Markdown(markdown_text)


# 定义模型参数
model_params = {
    "num_users": {
        "type": "SliderInt",
        "value": 100,
        "label": "用户数量",
        "min": 10,
        "max": 500,
        "step": 10
    },
    "Q_strength": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "算法个性化强度 (Q)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1
    },
    "P_strength": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "确认偏误强度 (P)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1
    },
    "learning_rate": {
        "type": "SliderFloat",
        "value": 0.05,
        "label": "信念更新速率",
        "min": 0.01,
        "max": 0.2,
        "step": 0.01
    },
    "content_pool_size": {
        "type": "SliderInt",
        "value": 1000,
        "label": "内容池大小",
        "min": 100,
        "max": 2000,
        "step": 100
    }
}


# 创建 SolaraViz 实例
page = SolaraViz(
    PlatformModel,
    model_params,
    name="信息茧房与确认偏误共谋模型"
)

# 添加自定义组件
page.components = [ModelInfo, belief_distribution_chart, polarization_chart, mean_belief_chart]
