"""
单次模拟分析脚本
用于快速测试和分析单个配置的模型运行
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from model import PlatformModel


def run_and_analyze(Q_strength=0.8, P_strength=0.8, num_users=100, steps=200):
    """
    运行单次模拟并生成详细分析
    
    Args:
        Q_strength: 算法个性化强度
        P_strength: 确认偏误强度
        num_users: 用户数量
        steps: 模拟步数
    """
    print("="*70)
    print(f"运行模拟: Q={Q_strength}, P={P_strength}")
    print("="*70)
    
    # 创建并运行模型
    model = PlatformModel(
        num_users=num_users,
        Q_strength=Q_strength,
        P_strength=P_strength,
        learning_rate=0.05,
        content_pool_size=1000
    )
    
    print(f"\n模拟 {steps} 个时间步...")
    for i in range(steps):
        model.step()
        if (i + 1) % 50 == 0:
            print(f"进度: {i+1}/{steps}")
    
    print("\n✓ 模拟完成！\n")
    
    # 获取数据
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    # 统计分析
    print("="*70)
    print("结果分析")
    print("="*70)
    
    initial_polarization = model_data['Polarization'].iloc[0]
    final_polarization = model_data['Polarization'].iloc[-1]
    polarization_increase = final_polarization - initial_polarization
    
    print(f"\n【极化程度】")
    print(f"  初始极化: {initial_polarization:.4f}")
    print(f"  最终极化: {final_polarization:.4f}")
    print(f"  增长量:   {polarization_increase:.4f}")
    print(f"  增长率:   {(polarization_increase/initial_polarization*100):.1f}%")
    
    initial_beliefs = agent_data.xs(0, level='Step')['Belief'].values
    final_beliefs = agent_data.xs(steps-1, level='Step')['Belief'].values
    
    print(f"\n【信念分布】")
    print(f"  初始均值: {np.mean(initial_beliefs):.3f}")
    print(f"  最终均值: {np.mean(final_beliefs):.3f}")
    print(f"  初始标准差: {np.std(initial_beliefs):.3f}")
    print(f"  最终标准差: {np.std(final_beliefs):.3f}")
    
    # 检测双峰分布（极化的标志）
    final_left = np.sum(final_beliefs < -0.3)
    final_center = np.sum((final_beliefs >= -0.3) & (final_beliefs <= 0.3))
    final_right = np.sum(final_beliefs > 0.3)
    
    print(f"\n【观点分布】")
    print(f"  左翼 (< -0.3):  {final_left} 人 ({final_left/num_users*100:.1f}%)")
    print(f"  中间 (-0.3~0.3): {final_center} 人 ({final_center/num_users*100:.1f}%)")
    print(f"  右翼 (> 0.3):   {final_right} 人 ({final_right/num_users*100:.1f}%)")
    
    # 可视化
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 极化趋势
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(model_data.index, model_data['Polarization'], 'r-', linewidth=2)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('极化程度 (方差)')
    ax1.set_title('极化程度随时间变化')
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均信念趋势
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(model_data.index, model_data['Mean_Belief'], 'b-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('平均信念')
    ax2.set_title('平均信念随时间变化')
    ax2.grid(True, alpha=0.3)
    
    # 3. 标准差趋势
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(model_data.index, model_data['Belief_Std'], 'g-', linewidth=2)
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('信念标准差')
    ax3.set_title('信念分散程度随时间变化')
    ax3.grid(True, alpha=0.3)
    
    # 4. 初始信念分布
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(initial_beliefs, bins=30, range=(-1, 1), 
             color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('信念值')
    ax4.set_ylabel('人数')
    ax4.set_title(f'初始信念分布 (t=0)\n方差={initial_polarization:.3f}')
    ax4.set_xlim(-1, 1)
    
    # 5. 最终信念分布
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(final_beliefs, bins=30, range=(-1, 1), 
             color='lightcoral', edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('信念值')
    ax5.set_ylabel('人数')
    ax5.set_title(f'最终信念分布 (t={steps})\n方差={final_polarization:.3f}')
    ax5.set_xlim(-1, 1)
    
    # 6. 对比分布
    ax6 = plt.subplot(2, 3, 6)
    bins = np.linspace(-1, 1, 31)
    ax6.hist(initial_beliefs, bins=bins, alpha=0.5, 
             label='初始', color='blue', density=True)
    ax6.hist(final_beliefs, bins=bins, alpha=0.5, 
             label='最终', color='red', density=True)
    ax6.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax6.set_xlabel('信念值')
    ax6.set_ylabel('密度')
    ax6.set_title('信念分布对比')
    ax6.legend()
    ax6.set_xlim(-1, 1)
    
    plt.suptitle(f'信息茧房模拟分析\nQ={Q_strength}, P={P_strength}, 用户={num_users}, 步数={steps}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'analysis_Q{Q_strength}_P{P_strength}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存至: {filename}")
    plt.show()
    
    print("="*70)
    
    return model, model_data, agent_data


if __name__ == "__main__":
    print("\n" + "="*70)
    print("信息茧房单次模拟分析")
    print("="*70)
    print("\n默认配置: 共谋场景 (高P高Q)")
    print("你可以修改代码中的参数来测试不同场景\n")
    
    # 运行分析
    # 可以修改这里的参数来测试不同场景
    model, model_data, agent_data = run_and_analyze(
        Q_strength=0.8,  # 算法个性化强度
        P_strength=0.8,  # 确认偏误强度
        num_users=100,   # 用户数量
        steps=200        # 模拟步数
    )
    
    print("\n提示: 尝试修改 Q_strength 和 P_strength 来探索不同场景!")
    print("  - 基线场景: Q=0.1, P=0.1")
    print("  - 选择性接触: Q=0.1, P=0.8")
    print("  - 信息茧房: Q=0.8, P=0.1")
    print("  - 共谋场景: Q=0.8, P=0.8")

