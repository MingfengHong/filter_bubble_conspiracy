"""
完全自定义的 Solara 可视化界面
避免使用 Mesa SolaraViz 的 bug
"""
import solara
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
from model import PlatformModel


# 全局状态
model = solara.reactive(None)
running = solara.reactive(False)
step_count = solara.reactive(0)

# 参数状态
num_users = solara.reactive(100)
Q_strength = solara.reactive(0.5)
P_strength = solara.reactive(0.5)
learning_rate = solara.reactive(0.05)
content_pool_size = solara.reactive(1000)


def reset_model():
    """重置模型"""
    model.value = PlatformModel(
        num_users=num_users.value,
        Q_strength=Q_strength.value,
        P_strength=P_strength.value,
        learning_rate=learning_rate.value,
        content_pool_size=content_pool_size.value
    )
    step_count.value = 0


def step_model():
    """运行一步"""
    if model.value is not None:
        model.value.step()
        step_count.value = model.value.steps


@solara.component
def ControlPanel():
    """控制面板"""
    with solara.Card("控制面板"):
        solara.SliderInt("用户数量", value=num_users, min=10, max=500, step=10)
        solara.SliderFloat("算法个性化强度 (Q)", value=Q_strength, min=0.0, max=1.0, step=0.1)
        solara.SliderFloat("确认偏误强度 (P)", value=P_strength, min=0.0, max=1.0, step=0.1)
        solara.SliderFloat("信念更新速率", value=learning_rate, min=0.01, max=0.2, step=0.01)
        solara.SliderInt("内容池大小", value=content_pool_size, min=100, max=2000, step=100)
        
        with solara.Row():
            solara.Button("重置模型", on_click=reset_model, color="primary")
            solara.Button("运行一步", on_click=step_model, disabled=model.value is None)


@solara.component
def ModelInfo():
    """模型信息"""
    if model.value is None:
        solara.Warning("请先点击'重置模型'创建模型")
        return
    
    # 判断场景
    Q = model.value.Q_strength
    P = model.value.P_strength
    
    if Q < 0.3 and P < 0.3:
        scenario = "🔷 基线场景 (低P低Q)"
        desc = "多元环境，开放心态"
        color = "primary"
    elif Q < 0.3 and P >= 0.6:
        scenario = "🟢 选择性接触 (高P低Q)"
        desc = "用户主动寻找认同"
        color = "success"
    elif Q >= 0.6 and P < 0.3:
        scenario = "🟡 信息茧房 (低P高Q)"
        desc = "算法限制信息接触"
        color = "warning"
    elif Q >= 0.6 and P >= 0.6:
        scenario = "🔴 共谋场景 (高P高Q)"
        desc = "算法与偏误共同作用 - 最强极化！"
        color = "error"
    else:
        scenario = "⚪ 中等强度场景"
        desc = "混合效应"
        color = "default"
    
    with solara.Card(f"{scenario}"):
        solara.Markdown(f"""
**{desc}**

---

**模型状态:**
- 当前步数: {model.value.steps}
- 用户数量: {model.value.num_users}
- Q 强度: {Q:.2f}
- P 强度: {P:.2f}
""")


@solara.component
def BeliefDistribution():
    """信念分布图"""
    if model.value is None or len(model.value.agents) == 0:
        solara.Info("等待模型初始化...")
        return
    
    beliefs = [agent.belief for agent in model.value.agents]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(beliefs, bins=20, range=(-1, 1), color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('信念值', fontsize=12)
    ax.set_ylabel('用户数量', fontsize=12)
    ax.set_title(f'信念分布 (步数: {model.value.steps})', fontsize=14, fontweight='bold')
    ax.set_xlim(-1, 1)
    
    # 统计信息
    mean_b = np.mean(beliefs)
    std_b = np.std(beliefs)
    var_b = np.var(beliefs)
    
    stats = f'均值: {mean_b:.3f}\n标准差: {std_b:.3f}\n极化: {var_b:.3f}'
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    solara.FigureMatplotlib(fig, dependencies=[step_count.value])
    plt.close(fig)


@solara.component
def PolarizationChart():
    """极化趋势图"""
    if model.value is None:
        solara.Info("等待模型初始化...")
        return
    
    model_data = model.value.datacollector.get_model_vars_dataframe()
    if len(model_data) == 0:
        solara.Info("开始运行以查看数据...")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model_data.index, model_data['Polarization'], 'r-', linewidth=2, label='极化程度')
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('极化程度 (方差)', fontsize=12)
    ax.set_title('极化程度随时间变化', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    solara.FigureMatplotlib(fig, dependencies=[step_count.value])
    plt.close(fig)


@solara.component
def MeanBeliefChart():
    """平均信念图"""
    if model.value is None:
        solara.Info("等待模型初始化...")
        return
    
    model_data = model.value.datacollector.get_model_vars_dataframe()
    if len(model_data) == 0:
        solara.Info("开始运行以查看数据...")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model_data.index, model_data['Mean_Belief'], 'b-', linewidth=2, label='平均信念')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('平均信念', fontsize=12)
    ax.set_title('平均信念随时间变化', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    solara.FigureMatplotlib(fig, dependencies=[step_count.value])
    plt.close(fig)


@solara.component
def Page():
    """主页面"""
    with solara.Column(style={"padding": "20px"}):
        solara.Markdown("# 信息茧房与确认偏误共谋模型")
        solara.Markdown("使用滑块调整参数，点击'重置模型'应用新参数，然后'运行一步'开始模拟")
        
        with solara.Columns([1, 2]):
            # 左侧：控制面板
            with solara.Column():
                ControlPanel()
                ModelInfo()
            
            # 右侧：可视化
            with solara.Column():
                with solara.Card("信念分布"):
                    BeliefDistribution()
                
                with solara.Card("极化趋势"):
                    PolarizationChart()
                
                with solara.Card("平均信念"):
                    MeanBeliefChart()
