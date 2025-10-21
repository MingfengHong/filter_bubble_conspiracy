"""
简单的可视化运行脚本（不依赖 Solara）
使用 matplotlib 实时更新显示
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation
import numpy as np
from model import PlatformModel

# 创建模型
print("=" * 60)
print("信息茧房与确认偏误共谋模型 - 简化版可视化")
print("=" * 60)
print("\n配置参数:")
print("- 用户数量: 100")
print("- Q (算法个性化强度): 0.8")
print("- P (确认偏误强度): 0.8")
print("- 场景: 共谋场景 (高P高Q)")
print("\n提示: 修改下方的参数可以测试不同场景")
print("=" * 60)

# ===== 在这里修改参数 =====
model = PlatformModel(
    num_users=100,
    Q_strength=0.8,  # 修改这里: 0.1=低, 0.8=高
    P_strength=0.8,  # 修改这里: 0.1=低, 0.8=高
    learning_rate=0.05,
    content_pool_size=1000
)
# ========================

# 创建图表
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'信息茧房模型 - Q={model.Q_strength}, P={model.P_strength}', 
             fontsize=16, fontweight='bold')

ax1 = plt.subplot(2, 3, 1)  # 信念分布
ax2 = plt.subplot(2, 3, 2)  # 极化趋势
ax3 = plt.subplot(2, 3, 3)  # 平均信念
ax4 = plt.subplot(2, 3, 4)  # 初始分布
ax5 = plt.subplot(2, 3, 5)  # 当前分布
ax6 = plt.subplot(2, 3, 6)  # 统计信息

# 初始信念
initial_beliefs = [agent.belief for agent in model.agents]
ax4.hist(initial_beliefs, bins=20, range=(-1, 1), color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('信念值')
ax4.set_ylabel('人数')
ax4.set_title('初始信念分布')
ax4.set_xlim(-1, 1)

# 用于存储历史数据
polarization_history = []
mean_history = []


def update(frame):
    """更新函数"""
    # 运行一步
    model.step()
    
    # 获取数据
    beliefs = [agent.belief for agent in model.agents]
    model_data = model.datacollector.get_model_vars_dataframe()
    
    polarization_history.append(model_data['Polarization'].iloc[-1])
    mean_history.append(model_data['Mean_Belief'].iloc[-1])
    
    # 清空并更新图表
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax5.clear()
    ax6.clear()
    
    # 1. 当前信念分布
    ax1.hist(beliefs, bins=20, range=(-1, 1), color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('信念值')
    ax1.set_ylabel('人数')
    ax1.set_title(f'当前信念分布 (步数: {model.steps})')
    ax1.set_xlim(-1, 1)
    
    # 2. 极化趋势
    ax2.plot(range(len(polarization_history)), polarization_history, 'r-', linewidth=2)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('极化程度 (方差)')
    ax2.set_title('极化程度随时间变化')
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均信念趋势
    ax3.plot(range(len(mean_history)), mean_history, 'b-', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('平均信念')
    ax3.set_title('平均信念随时间变化')
    ax3.grid(True, alpha=0.3)
    
    # 5. 最终分布对比
    bins = np.linspace(-1, 1, 21)
    ax5.hist(initial_beliefs, bins=bins, alpha=0.5, label='初始', color='blue', density=True)
    ax5.hist(beliefs, bins=bins, alpha=0.5, label='当前', color='red', density=True)
    ax5.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax5.set_xlabel('信念值')
    ax5.set_ylabel('密度')
    ax5.set_title('信念分布对比')
    ax5.legend()
    ax5.set_xlim(-1, 1)
    
    # 6. 统计信息
    ax6.axis('off')
    stats_text = f"""
    【模型参数】
    算法个性化强度 (Q): {model.Q_strength:.2f}
    确认偏误强度 (P): {model.P_strength:.2f}
    用户数量: {model.num_users}
    当前步数: {model.steps}
    
    【当前统计】
    极化程度: {np.var(beliefs):.4f}
    平均信念: {np.mean(beliefs):.3f}
    标准差: {np.std(beliefs):.3f}
    
    【分布情况】
    左翼 (< -0.3): {np.sum(np.array(beliefs) < -0.3)} 人
    中间 (-0.3~0.3): {np.sum((np.array(beliefs) >= -0.3) & (np.array(beliefs) <= 0.3))} 人
    右翼 (> 0.3): {np.sum(np.array(beliefs) > 0.3)} 人
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 停止条件
    if model.steps >= 200:
        print(f"\n模拟完成！运行了 {model.steps} 步")
        print(f"最终极化程度: {np.var(beliefs):.4f}")
        ani.event_source.stop()


print("\n开始模拟...")
print("关闭图表窗口将停止模拟\n")

# 创建动画
ani = FuncAnimation(fig, update, frames=200, interval=100, repeat=False)

plt.show()

print("\n模拟已结束")

