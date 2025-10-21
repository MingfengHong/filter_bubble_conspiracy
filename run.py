"""
启动 Mesa 可视化服务器 (Mesa 3.x + Solara)
运行此脚本后，在浏览器中访问显示的地址
"""
import subprocess
import sys

if __name__ == "__main__":
    print("=" * 60)
    print("信息茧房与确认偏误共谋模型")
    print("=" * 60)
    print("\n模型说明:")
    print("- Q (算法个性化强度): 控制推荐算法的个性化程度")
    print("- P (确认偏误强度): 控制用户选择性接触的倾向")
    print("\n实验建议:")
    print("1. 基线 (Q=0.1, P=0.1): 观察多元环境下的信念演化")
    print("2. 选择性接触 (Q=0.1, P=0.8): 观察用户主动寻找认同的效应")
    print("3. 信息茧房 (Q=0.8, P=0.1): 观察算法限制信息的效应")
    print("4. 共谋 (Q=0.8, P=0.8): 观察算法与偏误共同作用的最强效应")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60)
    print("\n正在启动可视化服务器...")
    print("服务器启动后，浏览器将自动打开")
    print("或手动访问显示的地址（通常是 http://localhost:8765）")
    print("=" * 60)
    print()
    
    # 使用 solara 运行应用
    try:
        subprocess.run([sys.executable, "-m", "solara", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n服务器已停止。")
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n如果上述命令失败，请尝试直接运行:")
        print("  solara run app.py")
