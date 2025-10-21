"""
2x2 实验设计实时对比
同时运行四种场景并实时可视化对比
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation
import numpy as np
from model import PlatformModel

print("=" * 70)
print("信息茧房 2x2 实验设计 - 实时对比")
print("=" * 70)
print("\n创建四个模型...")

# 创建四个模型 (2x2 设计)
# 使用更高的learning_rate让效果更明显
models = {
    '基线\n(低P低Q)': PlatformModel(num_users=100, Q_strength=0.1, P_strength=0.1, learning_rate=0.1),
    '选择性接触\n(高P低Q)': PlatformModel(num_users=100, Q_strength=0.1, P_strength=0.8, learning_rate=0.1),
    '信息茧房\n(低P高Q)': PlatformModel(num_users=100, Q_strength=0.8, P_strength=0.1, learning_rate=0.1),
    '共谋\n(高P高Q)': PlatformModel(num_users=100, Q_strength=0.8, P_strength=0.8, learning_rate=0.1)
}

# 存储初始信念分布
initial_beliefs = {}
for name, model in models.items():
    initial_beliefs[name] = [agent.belief for agent in model.agents]

print("✓ 模型创建完成")
print("\n场景说明:")
print("1. 基线 (Q=0.1, P=0.1) - 多元环境，开放心态")
print("2. 选择性接触 (Q=0.1, P=0.8) - 用户主动寻找认同")
print("3. 信息茧房 (Q=0.8, P=0.1) - 算法限制信息接触")
print("4. 共谋 (Q=0.8, P=0.8) - 算法与偏误共同作用")
print("\n开始模拟...")
print("关闭图表窗口将停止模拟\n")

# 创建图表布局 (3行4列)
fig = plt.figure(figsize=(20, 12))
fig.suptitle('信息茧房 2x2 实验设计：算法 (Q) vs 确认偏误 (P)', 
             fontsize=16, fontweight='bold')

# 为每个模型创建3个子图
axes = {}
for i, (name, model) in enumerate(models.items()):
    col = i  # 列：0-3
    
    # 第1行：当前信念分布
    axes[f'{name}_dist'] = plt.subplot(3, 4, col + 1)
    
    # 第2行：极化趋势
    axes[f'{name}_polar'] = plt.subplot(3, 4, col + 5)
    
    # 第3行：分布对比（初始vs当前）
    axes[f'{name}_compare'] = plt.subplot(3, 4, col + 9)

# 配置颜色
colors = {
    '基线\n(低P低Q)': 'blue',
    '选择性接触\n(高P低Q)': 'green',
    '信息茧房\n(低P高Q)': 'orange',
    '共谋\n(高P高Q)': 'red'
}


def update(frame):
    """更新函数"""
    # 运行每个模型一步
    for model in models.values():
        model.step()
    
    # 清空所有图表
    for ax in axes.values():
        ax.clear()
    
    # 收集所有模型的最大极化值，用于统一y轴
    max_polarization = 0
    for model in models.values():
        model_data = model.datacollector.get_model_vars_dataframe()
        if len(model_data) > 0:
            max_polarization = max(max_polarization, model_data['Polarization'].max())
    
    # 为每个模型更新图表
    for i, (name, model) in enumerate(models.items()):
        color = colors[name]
        beliefs = [agent.belief for agent in model.agents]
        model_data = model.datacollector.get_model_vars_dataframe()
        
        # 第1行：当前信念分布
        ax_dist = axes[f'{name}_dist']
        ax_dist.hist(beliefs, bins=20, range=(-1, 1), color=color, 
                    edgecolor='black', alpha=0.7)
        ax_dist.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_dist.set_xlabel('信念值', fontsize=9)
        ax_dist.set_ylabel('人数', fontsize=9)
        ax_dist.set_title(f'{name}\n步数: {model.steps}', fontsize=11, fontweight='bold')
        ax_dist.set_xlim(-1, 1)
        ax_dist.set_ylim(0, 40)
        
        # 添加统计信息
        polarization = np.var(beliefs)
        mean_b = np.mean(beliefs)
        stats = f'极化: {polarization:.3f}\n均值: {mean_b:.2f}'
        ax_dist.text(0.02, 0.98, stats, transform=ax_dist.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 第2行：极化趋势
        ax_polar = axes[f'{name}_polar']
        if len(model_data) > 0:
            ax_polar.plot(model_data.index, model_data['Polarization'], 
                         color=color, linewidth=2, label=f'当前: {model_data["Polarization"].iloc[-1]:.3f}')
            ax_polar.set_xlabel('时间步', fontsize=9)
            ax_polar.set_ylabel('极化程度', fontsize=9)
            ax_polar.set_title('极化趋势', fontsize=10)
            ax_polar.grid(True, alpha=0.3)
            ax_polar.legend(fontsize=8, loc='upper left')
            # 使用统一的y轴范围，便于对比
            y_max = max(max_polarization * 1.1, 0.05)
            ax_polar.set_ylim(0, y_max)
        
        # 第3行：初始vs当前分布对比
        ax_compare = axes[f'{name}_compare']
        bins = np.linspace(-1, 1, 21)
        ax_compare.hist(initial_beliefs[name], bins=bins, alpha=0.4, 
                       label='初始', color='gray', density=True)
        ax_compare.hist(beliefs, bins=bins, alpha=0.6, 
                       label='当前', color=color, density=True)
        ax_compare.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax_compare.set_xlabel('信念值', fontsize=9)
        ax_compare.set_ylabel('密度', fontsize=9)
        ax_compare.set_title('分布对比', fontsize=10)
        ax_compare.legend(fontsize=8)
        ax_compare.set_xlim(-1, 1)
        ax_compare.set_ylim(0, 3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 停止条件
    if frame >= 199:
        print(f"\n模拟完成！运行了 {frame+1} 步")
        print("\n最终极化程度对比:")
        for name, model in models.items():
            beliefs = [agent.belief for agent in model.agents]
            final_polar = np.var(beliefs)
            print(f"  {name.replace(chr(10), ' ')}: {final_polar:.4f}")
        ani.event_source.stop()


# 创建动画
ani = FuncAnimation(fig, update, frames=200, interval=100, repeat=False)

plt.show()

print("\n实验结束")
print("\n理论预测验证:")
print("极化程度排序应该是: 共谋 > 信息茧房 ≈ 选择性接触 > 基线")

