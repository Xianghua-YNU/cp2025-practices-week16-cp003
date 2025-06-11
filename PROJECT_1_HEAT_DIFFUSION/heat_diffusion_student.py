"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000     # 时间步数

x = np.linspace(0, L, Nx)

def basic_heat_diffusion():
    u = np.zeros((Nx, Nt))
    u[:, 0] = np.sin(np.pi * x)  # 初始条件

    r = D * dt / dx**2
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            u[i, n+1] = u[i, n] + r * (u[i+1, n] - 2*u[i, n] + u[i-1, n])
    return u

def analytical_solution(n_terms=100):
    """任务2: 解析解函数"""
    u = np.zeros((Nx, Nt))
    for n in range(Nt):
        t = n * dt
        for m in range(1, n_terms+1):
            coeff = (4 / (m * np.pi)) * (1 if m % 2 == 1 else 0)
            u[:, n] += coeff * np.sin(m * np.pi * x) * np.exp(-D * (m * np.pi)**2 * t)
    return u

def stability_analysis():
    """任务3: 数值解稳定性分析"""
    dx = 0.01
    dt = 0.6  # 故意选大步长，导致不稳定
    r = D * dt / (dx**2)
    print(f"任务3 - 稳定性参数 r = {r:.3f} (应 <= 0.5)")

    Nx = int(L / dx) + 1
    Nt = 2000

    u = np.zeros((Nx, Nt))
    u[:, 0] = 100
    u[0, :] = 0
    u[-1, :] = 0

    for j in range(Nt - 1):
        u[1:-1, j+1] = (1 - 2*r) * u[1:-1, j] + r * (u[2:, j] + u[:-2, j])

    plot_3d_solution(u, dx, dt, Nt, title='任务3：不稳定解 (r > 0.5)')


def different_initial_condition():
    """任务4: 不同初始条件模拟"""
    u = np.zeros((Nx, Nt))
    u[:, 0] = np.exp(-((x - 0.5)**2) / 0.01)  # 高斯初始分布

    r = D * dt / dx**2
    for n in range(0, Nt - 1):
        u[1:-1, n+1] = u[1:-1, n] + r * (u[2:, n] - 2*u[1:-1, n] + u[:-2, n])
    return u

def heat_diffusion_with_cooling():
    """任务5: 包含牛顿冷却定律的热传导"""
    h = 5.0     # 对流换热系数
    T_inf = 0   # 环境温度

    u = np.zeros((Nx, Nt))
    u[:, 0] = np.sin(np.pi * x)

    r = D * dt / dx**2
    beta = h * dt / (C * rho)
    for n in range(0, Nt - 1):
        u[1:-1, n+1] = u[1:-1, n] + r * (u[2:, n] - 2*u[1:-1, n] + u[:-2, n]) - beta * (u[1:-1, n] - T_inf)
    return u


def plot_3d_solution(u, dx, dt, Nt, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(0, u.shape[0]) * dx
    T = np.arange(0, Nt) * dt
    X, T = np.meshgrid(X, T)
    ax.plot_surface(X, T, u.T, cmap='hot')
    ax.set_xlabel('位置 x (m)')
    ax.set_ylabel('时间 t (s)')
    ax.set_zlabel('温度 T')
    ax.set_title(title)
    plt.show()

    
if __name__ == "__main__":
    print("=== 铝棒热传导问题参考答案 ===")
    print("1. 基本热传导模拟")
    u = basic_heat_diffusion()
    plot_3d_solution(u, dx, dt, Nt, title='Task 1: Heat Diffusion Solution')

    print("\n2. 解析解")
    s = analytical_solution()
    plot_3d_solution(s, dx, dt, Nt, title='Analytical Solution')

    print("\n3. 数值解稳定性分析")
    stability_analysis()
    
    print("\n4. 不同初始条件模拟")
    different_initial_condition()
    
    print("\n5. 包含牛顿冷却定律的热传导")
    heat_diffusion_with_cooling()
