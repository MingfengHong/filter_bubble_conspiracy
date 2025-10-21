"""
批量实验脚本
系统地比较不同 P 和 Q 组合下的极化效应
实现 2x2 实验设计以解耦算法与偏误的作用
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from model import PlatformModel
import pandas as pd


def run_single_experiment(Q_strength, P_strength, num_users=100, steps=200):
    """
    运行单个实验
    
    Args:
        Q_strength: 算法个性化强度
        P_strength: 确认偏误强度
        num_users: 用户数量
        steps: 模拟步数
        
    Returns:
        模型实例和数据
    """
    print(f"运行实验: Q={Q_strength:.1f}, P={P_strength:.1f}")
    
    model = PlatformModel(
        num_users=num_users,
        Q_strength=Q_strength,
        P_strength=P_strength,
        learning_rate=0.05,
        content_pool_size=1000
    )
    
    # 运行模拟
    for i in range(steps):
        model.step()
        if (i + 1) % 50 == 0:
            print(f"  步数 {i+1}/{steps}")
    
    return model


def plot_2x2_experiment(results, steps=200):
    """
    绘制 2x2 实验设计的结果
    
    Args:
        results: 实验结果字典
        steps: 模拟步数
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('信息茧房 2x2 实验设计：算法 (Q) vs 确认偏误 (P)', 
                 fontsize=16, fontweight='bold')
    
    scenarios = [
        ('低P低Q', 0.1, 0.1, 0, 0),
        ('高P低Q', 0.1, 0.8, 0, 1),
        ('低P高Q', 0.8, 0.1, 1, 0),
        ('高P高Q', 0.8, 0.8, 1, 1)
    ]
    
    for name, Q, P, row, col in scenarios:
        ax = axes[row, col]
        key = f"Q{Q}_P{P}"
        
        if key not in results:
            continue
        
        model = results[key]
        
        # 获取数据
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()
        
        # 子图1: 极化趋势
        ax_polar = ax
        ax_polar.plot(model_data.index, model_data['Polarization'], 
                     'r-', linewidth=2, label='极化程度')
        ax_polar.set_xlabel('时间步', fontsize=10)
        ax_polar.set_ylabel('极化程度 (方差)', color='r', fontsize=10)
        ax_polar.tick_params(axis='y', labelcolor='r')
        ax_polar.grid(True, alpha=0.3)
        
        # 子图2: 信念分布（初始 vs 最终）
        ax_hist = ax.twinx()
        
        # 获取初始和最终信念
        initial_beliefs = agent_data.xs(0, level='Step')['Belief'].values
        final_beliefs = agent_data.xs(steps-1, level='Step')['Belief'].values
        
        # 绘制分布对比
        bins = np.linspace(-1, 1, 21)
        ax_hist.hist(initial_beliefs, bins=bins, alpha=0.3, 
                    color='blue', label='初始分布', density=True)
        ax_hist.hist(final_beliefs, bins=bins, alpha=0.5, 
                    color='green', label='最终分布', density=True)
        ax_hist.set_ylabel('密度', color='g', fontsize=10)
        ax_hist.tick_params(axis='y', labelcolor='g')
        ax_hist.set_ylim(0, 3)
        
        # 标题和图例
        final_polarization = model_data['Polarization'].iloc[-1]
        ax.set_title(f'{name}\nQ={Q}, P={P}\n最终极化={final_polarization:.3f}', 
                    fontsize=12, fontweight='bold')
        
        # 合并图例
        lines1, labels1 = ax_polar.get_legend_handles_labels()
        lines2, labels2 = ax_hist.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('experiment_2x2_results.png', dpi=300, bbox_inches='tight')
    print("\n结果已保存至: experiment_2x2_results.png")
    plt.show()


def plot_polarization_comparison(results, steps=200):
    """
    绘制四种场景的极化趋势对比图
    """
    plt.figure(figsize=(12, 6))
    
    scenarios = [
        ('基线 (低P低Q)', 0.1, 0.1, 'blue', '--'),
        ('选择性接触 (高P低Q)', 0.1, 0.8, 'green', '-.'),
        ('信息茧房 (低P高Q)', 0.8, 0.1, 'orange', ':'),
        ('共谋 (高P高Q)', 0.8, 0.8, 'red', '-')
    ]
    
    for name, Q, P, color, linestyle in scenarios:
        key = f"Q{Q}_P{P}"
        if key in results:
            model_data = results[key].datacollector.get_model_vars_dataframe()
            plt.plot(model_data.index, model_data['Polarization'], 
                    label=name, color=color, linestyle=linestyle, linewidth=2)
    
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('极化程度 (方差)', fontsize=12)
    plt.title('四种场景下的极化趋势对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('polarization_comparison.png', dpi=300, bbox_inches='tight')
    print("结果已保存至: polarization_comparison.png")
    plt.show()


def create_summary_table(results):
    """
    创建实验结果摘要表
    """
    summary_data = []
    
    scenarios = [
        ('低P低Q (基线)', 0.1, 0.1),
        ('高P低Q (选择性接触)', 0.1, 0.8),
        ('低P高Q (信息茧房)', 0.8, 0.1),
        ('高P高Q (共谋)', 0.8, 0.8)
    ]
    
    for name, Q, P in scenarios:
        key = f"Q{Q}_P{P}"
        if key in results:
            model_data = results[key].datacollector.get_model_vars_dataframe()
            
            initial_polarization = model_data['Polarization'].iloc[0]
            final_polarization = model_data['Polarization'].iloc[-1]
            polarization_increase = final_polarization - initial_polarization
            
            summary_data.append({
                '场景': name,
                'Q强度': Q,
                'P强度': P,
                '初始极化': f"{initial_polarization:.4f}",
                '最终极化': f"{final_polarization:.4f}",
                '极化增量': f"{polarization_increase:.4f}",
                '增长率': f"{(polarization_increase/initial_polarization*100):.1f}%"
            })
    
    df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    df.to_csv('experiment_summary.csv', index=False, encoding='utf-8-sig')
    print("\n摘要表已保存至: experiment_summary.csv")
    
    return df


def main():
    """
    主实验流程
    """
    print("="*80)
    print("信息茧房与确认偏误共谋：2x2 实验设计")
    print("="*80)
    print("\n本实验将系统地比较四种场景:")
    print("1. 低P低Q (基线): 多元环境，开放心态")
    print("2. 高P低Q (选择性接触): 用户主动寻找认同")
    print("3. 低P高Q (信息茧房): 算法限制信息接触")
    print("4. 高P高Q (共谋): 算法与偏误共同作用")
    print("\n每个实验将运行 200 步，模拟 100 个用户...")
    print("="*80 + "\n")
    
    # 实验参数
    num_users = 100
    steps = 200
    
    # 运行 2x2 实验
    results = {}
    
    for Q in [0.1, 0.8]:  # 低Q, 高Q
        for P in [0.1, 0.8]:  # 低P, 高P
            key = f"Q{Q}_P{P}"
            results[key] = run_single_experiment(Q, P, num_users, steps)
            print(f"✓ 完成: Q={Q}, P={P}\n")
    
    # 生成可视化和报告
    print("\n生成结果图表...")
    plot_2x2_experiment(results, steps)
    plot_polarization_comparison(results, steps)
    create_summary_table(results)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print("\n主要发现:")
    print("- 查看 experiment_2x2_results.png 了解各场景详情")
    print("- 查看 polarization_comparison.png 对比极化趋势")
    print("- 查看 experiment_summary.csv 获取定量结果")


if __name__ == "__main__":
    main()

