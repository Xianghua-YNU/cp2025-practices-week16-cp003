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
    u = np.zeros((Nx, Nt))
    for n in range(Nt):
        t = n * dt
        for m in range(1, n_terms+1):
            coeff = (4 / (m * np.pi)) * (1 if m % 2 == 1 else 0)
            u[:, n] += coeff * np.sin(m * np.pi * x) * np.exp(-D * (m * np.pi)**2 * t)
    return u

def stability_analysis():
    r = D * dt / dx**2
    if r <= 0.5:
        print(f"稳定：r = {r:.3f} <= 0.5")
    else:
        print(f"不稳定：r = {r:.3f} > 0.5")

def different_initial_condition():
    u = np.zeros((Nx, Nt))
    u[:, 0] = np.exp(-((x - 0.5)**2) / 0.01)  # 高斯初始条件

    r = D * dt / dx**2
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            u[i, n+1] = u[i, n] + r * (u[i+1, n] - 2*u[i, n] + u[i-1, n])
    return u

def heat_diffusion_with_cooling():
    h = 5.0   # 对流换热系数
    T_inf = 0 # 环境温度
    u = np.zeros((Nx, Nt))
    u[:, 0] = np.sin(np.pi * x)

    r = D * dt / dx**2
    beta = h * dt / (C * rho)
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            u[i, n+1] = u[i, n] + r * (u[i+1, n] - 2*u[i, n] + u[i-1, n]) - beta * (u[i, n] - T_inf)
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
    print("=== 铝棒热传导问题学生实现 ===")
    stability_analysis()

    u1 = basic_heat_diffusion()
    plot_3d_solution(u1, dx, dt, Nt, "任务1：基本热传导模拟")

    u2 = analytical_solution()
    plot_3d_solution(u2, dx, dt, Nt, "任务2：解析解")

    u3 = different_initial_condition()
    plot_3d_solution(u3, dx, dt, Nt, "任务4：不同初始条件")

    u4 = heat_diffusion_with_cooling()
    plot_3d_solution(u4, dx, dt, Nt, "任务5：包含冷却效应的热传导")
