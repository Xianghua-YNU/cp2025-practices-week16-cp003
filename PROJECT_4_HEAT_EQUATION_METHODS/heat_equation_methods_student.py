#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg#!/usr/bin/env python3
"""
一维热传导方程多方法求解器（过程式版本）
文件：heat_equation_solver.py

实现四种数值方法求解一维热传导方程：
1. 显式有限差分法（FTCS）
2. 隐式有限差分法（BTCS）
3. Crank-Nicolson方法
4. scipy.integrate.solve_ivp方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

# ---------------------
# 优化的中文字体配置（Windows优先）
# ---------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# ---------------------
# 初始条件设置函数
# ---------------------
def set_initial_condition(L, nx):
    """
    设置初始温度分布：在10 ≤ x ≤ 11区域内温度为1，其余为0，并应用边界条件
    
    参数:
        L (float): 求解区域长度
        nx (int): 空间网格点数
        
    返回:
        np.ndarray: 初始温度分布数组
    """
    x = np.linspace(0, L, nx)
    u0 = np.zeros(nx)
    # 定义温度为1的区域
    mask = (x >= 10) & (x <= 11)
    u0[mask] = 1.0
    # 应用边界条件（两端温度为0）
    u0[0] = 0.0
    u0[-1] = 0.0
    return u0, x

# ---------------------
# 显式有限差分法（FTCS）
# ---------------------
def solve_explicit(L, alpha, nx, T_final, dt, plot_times=None):
    """
    使用显式有限差分法（FTCS）求解热传导方程
    
    参数:
        L (float): 求解区域长度
        alpha (float): 热扩散系数
        nx (int): 空间网格点数
        T_final (float): 最终模拟时间
        dt (float): 时间步长
        plot_times (list): 需要保存结果的时间点列表
        
    返回:
        dict: 包含求解结果的字典，包括时间点、温度分布、计算时间等
    """
    if plot_times is None:
        plot_times = [0, 1, 5, 15, 25]
    
    # 计算空间步长
    dx = L / (nx - 1)
    # 计算稳定性参数（傅里叶数）
    r = alpha * dt / (dx**2)
    # 稳定性检查
    if r > 0.5:
        print(f"警告：显式方法稳定性条件被违反！r = {r:.4f} > 0.5")
        print(f"建议将时间步长减小到 < {0.5 * dx**2 / alpha:.6f}")
    
    # 获取初始条件
    u, x = set_initial_condition(L, nx)
    t = 0.0
    # 计算总时间步数
    nt = int(T_final / dt) + 1
    
    # 存储结果的字典
    results = {'times': [], 'solutions': [], 'method': '显式有限差分法（FTCS）'}
    
    # 存储初始条件
    if 0 in plot_times:
        results['times'].append(0.0)
        results['solutions'].append(u.copy())
    
    start_time = time.time()  # 开始计时
    
    # 时间步进循环
    for n in range(1, nt):
        # 使用拉普拉斯算子计算二阶空间导数
        du_dt = r * laplace(u)
        # 更新温度分布
        u += du_dt
        # 应用边界条件
        u[0] = 0.0
        u[-1] = 0.0
        
        t = n * dt  # 更新当前时间
        
        # 在指定时间点存储结果
        for plot_time in plot_times:
            if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                results['times'].append(t)
                results['solutions'].append(u.copy())
    
    results['computation_time'] = time.time() - start_time  # 计算耗时
    results['stability_parameter'] = r  # 存储稳定性参数
    results['x'] = x  # 添加空间坐标
    
    return results

# ---------------------
# 隐式有限差分法（BTCS）
# ---------------------
def solve_implicit(L, alpha, nx, T_final, dt, plot_times=None):
    """
    使用隐式有限差分法（BTCS）求解热传导方程
    
    参数:
        L (float): 求解区域长度
        alpha (float): 热扩散系数
        nx (int): 空间网格点数
        T_final (float): 最终模拟时间
        dt (float): 时间步长
        plot_times (list): 需要保存结果的时间点列表
        
    返回:
        dict: 包含求解结果的字典，包括时间点、温度分布、计算时间等
    """
    if plot_times is None:
        plot_times = [0, 1, 5, 15, 25]
    
    # 计算空间步长
    dx = L / (nx - 1)
    # 计算稳定性参数
    r = alpha * dt / (dx**2)
    # 计算总时间步数
    nt = int(T_final / dt) + 1
    
    # 获取初始条件
    u, x = set_initial_condition(L, nx)
    # 内部节点数（除去两端边界）
    num_internal = nx - 2
    
    # 构建三对角矩阵（用于隐式方法求解）
    banded_matrix = np.zeros((3, num_internal))
    banded_matrix[0, 1:] = -r      # 上对角线
    banded_matrix[1, :] = 1 + 2*r  # 主对角线
    banded_matrix[2, :-1] = -r     # 下对角线
    
    # 存储结果的字典
    results = {'times': [], 'solutions': [], 'method': '隐式有限差分法（BTCS）'}
    
    # 存储初始条件
    if 0 in plot_times:
        results['times'].append(0.0)
        results['solutions'].append(u.copy())
    
    start_time = time.time()  # 开始计时
    
    # 时间步进循环
    for n in range(1, nt):
        # 构建右侧向量（内部节点当前时间步的值）
        rhs = u[1:-1].copy()
        # 求解三对角线性方程组
        u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix, rhs)
        # 更新内部节点的温度值
        u[1:-1] = u_internal_new
        # 应用边界条件
        u[0] = 0.0
        u[-1] = 0.0
        
        t = n * dt  # 更新当前时间
        
        # 在指定时间点存储结果
        for plot_time in plot_times:
            if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                results['times'].append(t)
                results['solutions'].append(u.copy())
    
    results['computation_time'] = time.time() - start_time  # 计算耗时
    results['stability_parameter'] = r  # 存储稳定性参数
    results['x'] = x  # 添加空间坐标
    
    return results

# ---------------------
# Crank-Nicolson方法
# ---------------------
def solve_crank_nicolson(L, alpha, nx, T_final, dt, plot_times=None):
    """
    使用Crank-Nicolson方法求解热传导方程
    
    参数:
        L (float): 求解区域长度
        alpha (float): 热扩散系数
        nx (int): 空间网格点数
        T_final (float): 最终模拟时间
        dt (float): 时间步长
        plot_times (list): 需要保存结果的时间点列表
        
    返回:
        dict: 包含求解结果的字典，包括时间点、温度分布、计算时间等
    """
    if plot_times is None:
        plot_times = [0, 1, 5, 15, 25]
    
    # 计算空间步长
    dx = L / (nx - 1)
    # 计算稳定性参数
    r = alpha * dt / (dx**2)
    # 计算总时间步数
    nt = int(T_final / dt) + 1
    
    # 获取初始条件
    u, x = set_initial_condition(L, nx)
    # 内部节点数
    num_internal = nx - 2
    
    # 构建Crank-Nicolson方法的三对角矩阵
    banded_matrix_A = np.zeros((3, num_internal))
    banded_matrix_A[0, 1:] = -r/2    # 上对角线
    banded_matrix_A[1, :] = 1 + r    # 主对角线
    banded_matrix_A[2, :-1] = -r/2   # 下对角线
    
    # 存储结果的字典
    results = {'times': [], 'solutions': [], 'method': 'Crank-Nicolson方法'}
    
    # 存储初始条件
    if 0 in plot_times:
        results['times'].append(0.0)
        results['solutions'].append(u.copy())
    
    start_time = time.time()  # 开始计时
    
    # 时间步进循环
    for n in range(1, nt):
        # 构建右侧向量（包含当前时间和未来时间的混合项）
        u_internal = u[1:-1]
        rhs = (r/2) * u[:-2] + (1 - r) * u_internal + (r/2) * u[2:]
        # 求解三对角线性方程组
        u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix_A, rhs)
        # 更新内部节点的温度值
        u[1:-1] = u_internal_new
        # 应用边界条件
        u[0] = 0.0
        u[-1] = 0.0
        
        t = n * dt  # 更新当前时间
        
        # 在指定时间点存储结果
        for plot_time in plot_times:
            if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                results['times'].append(t)
                results['solutions'].append(u.copy())
    
    results['computation_time'] = time.time() - start_time  # 计算耗时
    results['stability_parameter'] = r  # 存储稳定性参数
    results['x'] = x  # 添加空间坐标
    
    return results

# ---------------------
# solve_ivp方法
# ---------------------
def heat_equation_ode(t, u_internal, L, alpha, nx):
    """
    定义用于solve_ivp的ODE系统
    
    参数:
        t (float): 当前时间
        u_internal (np.ndarray): 内部节点温度
        L (float): 求解区域长度
        alpha (float): 热扩散系数
        nx (int): 空间网格点数
        
    返回:
        np.ndarray: 内部节点温度的时间导数
    """
    # 重构完整温度分布（包含边界条件）
    dx = L / (nx - 1)
    u_full = np.concatenate(([0.0], u_internal, [0.0]))
    # 计算二阶空间导数（使用拉普拉斯算子）
    d2u_dx2 = laplace(u_full) / (dx**2)
    # 返回内部节点的时间导数
    return alpha * d2u_dx2[1:-1]

def solve_with_solve_ivp(L, alpha, nx, T_final, method='BDF', plot_times=None):
    """
    使用scipy.integrate.solve_ivp求解热传导方程
    
    参数:
        L (float): 求解区域长度
        alpha (float): 热扩散系数
        nx (int): 空间网格点数
        T_final (float): 最终模拟时间
        method (str): 积分方法
        plot_times (list): 需要保存结果的时间点列表
        
    返回:
        dict: 包含求解结果的字典，包括时间点、温度分布、计算时间等
    """
    if plot_times is None:
        plot_times = [0, 1, 5, 15, 25]
    
    # 获取初始条件
    u0, x = set_initial_condition(L, nx)
    # 内部节点的初始温度（除去边界）
    u0_internal = u0[1:-1]
    
    start_time = time.time()  # 开始计时
    
    # 求解ODE系统
    sol = solve_ivp(
        fun=lambda t, u: heat_equation_ode(t, u, L, alpha, nx),
        t_span=(0, T_final),
        y0=u0_internal,
        method=method,
        t_eval=plot_times,
        rtol=1e-8,
        atol=1e-10
    )
    
    computation_time = time.time() - start_time  # 计算耗时
    
    # 重构完整解（包含边界条件）
    results = {
        'times': sol.t.tolist(),
        'solutions': [],
        'method': f'solve_ivp ({method})',
        'computation_time': computation_time,
        'x': x  # 添加空间坐标
    }
    
    for i in range(len(sol.t)):
        u_full = np.concatenate(([0.0], sol.y[:, i], [0.0]))
        results['solutions'].append(u_full)
    
    return results

# ---------------------
# 方法比较函数
# ---------------------
def compare_methods(L=20.0, alpha=10.0, nx=21, T_final=25.0,
                   dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5,
                   ivp_method='BDF', plot_times=None):
    """
    比较四种数值方法的求解结果
    
    参数:
        L (float): 求解区域长度
        alpha (float): 热扩散系数
        nx (int): 空间网格点数
        T_final (float): 最终模拟时间
        dt_explicit (float): 显式方法时间步长
        dt_implicit (float): 隐式方法时间步长
        dt_cn (float): Crank-Nicolson方法时间步长
        ivp_method (str): solve_ivp使用的方法
        plot_times (list): 需要保存结果的时间点列表
        
    返回:
        dict: 包含所有方法求解结果的字典
    """
    if plot_times is None:
        plot_times = [0, 1, 5, 15, 25]
    
    print("使用四种不同方法求解热传导方程...")
    print(f"求解区域: [0, {L}], 网格点数: {nx}, 最终时间: {T_final}")
    print(f"热扩散系数: {alpha}")
    print("-" * 60)
    
    methods_results = {}  # 存储所有方法的结果
    
    # 显式方法
    print("1. 显式有限差分法（FTCS）...")
    methods_results['explicit'] = solve_explicit(
        L, alpha, nx, T_final, dt_explicit, plot_times
    )
    print(f"   计算时间: {methods_results['explicit']['computation_time']:.4f} 秒")
    print(f"   稳定性参数 r: {methods_results['explicit']['stability_parameter']:.4f}")
    
    # 隐式方法
    print("2. 隐式有限差分法（BTCS）...")
    methods_results['implicit'] = solve_implicit(
        L, alpha, nx, T_final, dt_implicit, plot_times
    )
    print(f"   计算时间: {methods_results['implicit']['computation_time']:.4f} 秒")
    print(f"   稳定性参数 r: {methods_results['implicit']['stability_parameter']:.4f}")
    
    # Crank-Nicolson方法
    print("3. Crank-Nicolson方法...")
    methods_results['crank_nicolson'] = solve_crank_nicolson(
        L, alpha, nx, T_final, dt_cn, plot_times
    )
    print(f"   计算时间: {methods_results['crank_nicolson']['computation_time']:.4f} 秒")
    print(f"   稳定性参数 r: {methods_results['crank_nicolson']['stability_parameter']:.4f}")
    
    # solve_ivp方法
    print(f"4. solve_ivp方法 ({ivp_method})...")
    methods_results['solve_ivp'] = solve_with_solve_ivp(
        L, alpha, nx, T_final, ivp_method, plot_times
    )
    print(f"   计算时间: {methods_results['solve_ivp']['computation_time']:.4f} 秒")
    
    print("-" * 60)
    print("所有方法求解完成！")
    
    return methods_results

# ---------------------
# 结果可视化函数
# ---------------------
def plot_comparison(methods_results, L, save_figure=False, filename='heat_equation_comparison.png'):
    """
    绘制不同方法的求解结果对比图
    
    参数:
        methods_results (dict): 包含所有方法结果的字典
        L (float): 求解区域长度
        save_figure (bool): 是否保存图片
        filename (str): 保存图片的文件名
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    method_names = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
    # 确保颜色数量与时间点数量匹配（5个时间点）
    colors = ['blue', 'red', 'green', 'orange', 'purple']  
    
    for idx, method_name in enumerate(method_names):
        ax = axes[idx]
        results = methods_results[method_name]
        
        # 绘制不同时间点的温度分布
        for i, (t, u) in enumerate(zip(results['times'], results['solutions'])):
            ax.plot(results['x'], u, color=colors[i], label=f'时间 t = {t:.1f}', linewidth=2)
        
        ax.set_title(f"{results['method']}\n(计算时间: {results['computation_time']:.4f} 秒)")
        ax.set_xlabel('位置 x')
        ax.set_ylabel('温度 u(x,t)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, L)
        ax.set_ylim(-0.1, 1.1)  # 扩展Y轴范围以便观察
        
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为 {filename}")
    
    plt.show()

# ---------------------
# 误差分析函数
# ---------------------
def analyze_accuracy(methods_results, reference_method='solve_ivp'):
    """
    分析不同方法的求解精度
    
    参数:
        methods_results (dict): 包含所有方法结果的字典
        reference_method (str): 作为参考的方法
        
    返回:
        dict: 包含各方法误差信息的字典
    """
    if reference_method not in methods_results:
        raise ValueError(f"参考方法 '{reference_method}' 未在结果中找到")
    
    reference = methods_results[reference_method]
    accuracy_results = {}
    
    print(f"\n精度分析（参考方法: {reference['method']}）")
    print("-" * 50)
    
    for method_name, results in methods_results.items():
        if method_name == reference_method:
            continue
            
        errors = []
        # 计算每个时间点的L2范数误差
        for i, (ref_sol, test_sol) in enumerate(zip(reference['solutions'], results['solutions'])):
            if i < len(results['solutions']):
                error = np.linalg.norm(ref_sol - test_sol, ord=2)
                errors.append(error)
        
        # 计算最大误差和平均误差
        max_error = max(errors) if errors else 0
        avg_error = np.mean(errors) if errors else 0
        
        accuracy_results[method_name] = {
            'max_error': max_error,
            'avg_error': avg_error,
            'errors': errors
        }
        
        print(f"{results['method']:25} - 最大误差: {max_error:.2e}, 平均误差: {avg_error:.2e}")
    
    return accuracy_results

# ---------------------
# 主函数
# ---------------------
def main():
    """主函数：协调各功能函数完成热传导方程的求解、比较和分析"""
    # 问题参数设置
    L = 20.0         # 求解区域长度
    alpha = 10.0     # 热扩散系数
    nx = 21          # 空间网格点数
    T_final = 25.0   # 最终模拟时间
    
    # 定义需要保存结果的时间点
    plot_times = [0, 1, 5, 15, 25]
    
    print(f"开始求解一维热传导方程，区域长度 L = {L}, 热扩散系数 alpha = {alpha}")
    print(f"空间网格点数 nx = {nx}, 最终模拟时间 T_final = {T_final}")
    print("-" * 80)
    
    # 比较四种方法的求解结果
    results = compare_methods(
        L=L, alpha=alpha, nx=nx, T_final=T_final,
        dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5,
        ivp_method='BDF', plot_times=plot_times
    )
    
    # 绘制结果对比图
    plot_comparison(results, L, save_figure=True)
    
    # 分析各方法的精度
    accuracy = analyze_accuracy(results, reference_method='solve_ivp')
    
    print("\n计算完成！")
    return results, accuracy

if __name__ == "__main__":
    # 执行主函数
    results, accuracy = main()

import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        # TODO: 创建零数组
        # TODO: 设置初始条件（10 <= x <= 11 区域为1）
        # TODO: 应用边界条件
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 显式差分法直接从当前时刻计算下一时刻的解
        数值方法: 使用scipy.ndimage.laplace计算空间二阶导数
        稳定性条件: r = alpha * dt / dx² <= 0.5
        
        实现步骤:
        1. 检查稳定性条件
        2. 初始化解数组和时间
        3. 时间步进循环
        4. 使用laplace算子计算空间导数
        5. 更新解并应用边界条件
        6. 存储指定时间点的解
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 计算稳定性参数 r = alpha * dt / dx²
        # TODO: 检查稳定性条件 r <= 0.5
        # TODO: 初始化解数组和时间变量
        # TODO: 创建结果存储字典
        # TODO: 存储初始条件
        # TODO: 时间步进循环
        #   - 使用 laplace(u) 计算空间二阶导数
        #   - 更新解：u += r * laplace(u)
        #   - 应用边界条件
        #   - 在指定时间点存储解
        # TODO: 返回结果字典
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 隐式差分法在下一时刻求解线性方程组
        数值方法: 构建三对角矩阵系统并求解
        优势: 无条件稳定，可以使用较大时间步长
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建三对角系数矩阵
        3. 时间步进循环
        4. 构建右端项
        5. 求解线性系统
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 计算扩散数 r
        # TODO: 构建三对角矩阵（内部节点）
        #   - 上对角线：-r
        #   - 主对角线：1 + 2r
        #   - 下对角线：-r
        # TODO: 初始化解数组和结果存储
        # TODO: 时间步进循环
        #   - 构建右端项（内部节点）
        #   - 使用 scipy.linalg.solve_banded 求解
        #   - 更新解并应用边界条件
        # TODO: 返回结果字典
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: Crank-Nicolson方法结合显式和隐式格式
        数值方法: 时间上二阶精度，无条件稳定
        优势: 高精度且稳定性好
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建左端矩阵 A
        3. 时间步进循环
        4. 构建右端向量
        5. 求解线性系统 A * u^{n+1} = rhs
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 计算扩散数 r
        # TODO: 构建左端矩阵 A（内部节点）
        #   - 上对角线：-r/2
        #   - 主对角线：1 + r
        #   - 下对角线：-r/2
        # TODO: 初始化解数组和结果存储
        # TODO: 时间步进循环
        #   - 构建右端向量：(r/2)*u[:-2] + (1-r)*u[1:-1] + (r/2)*u[2:]
        #   - 求解线性系统
        #   - 更新解并应用边界条件
        # TODO: 返回结果字典
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        
        参数:
            t (float): 当前时间
            u_internal (np.ndarray): 内部节点温度
            
        返回:
            np.ndarray: 内部节点的时间导数
            
        物理背景: 将PDE转化为ODE系统
        数值方法: 使用laplace算子计算空间导数
        
        实现步骤:
        1. 重构包含边界条件的完整解
        2. 使用laplace计算二阶导数
        3. 返回内部节点的导数
        """
        # TODO: 重构完整解向量（包含边界条件）
        # TODO: 使用 laplace(u_full) / dx² 计算二阶导数
        # TODO: 返回内部节点的时间导数：alpha * d²u/dx²
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        
        参数:
            method (str): 积分方法（'RK45', 'BDF', 'Radau'等）
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 将PDE转化为ODE系统求解
        数值方法: 使用高精度ODE求解器
        优势: 自适应步长，高精度
        
        实现步骤:
        1. 提取内部节点初始条件
        2. 调用solve_ivp求解ODE系统
        3. 重构包含边界条件的完整解
        4. 返回结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 提取内部节点初始条件
        # TODO: 调用 solve_ivp 求解
        #   - fun: self._heat_equation_ode
        #   - t_span: (0, T_final)
        #   - y0: 内部节点初始条件
        #   - method: 指定的积分方法
        #   - t_eval: plot_times
        # TODO: 重构包含边界条件的完整解
        # TODO: 返回结果字典
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        
        参数:
            dt_explicit (float): 显式方法时间步长
            dt_implicit (float): 隐式方法时间步长
            dt_cn (float): Crank-Nicolson方法时间步长
            ivp_method (str): solve_ivp积分方法
            plot_times (list): 比较时间点
            
        返回:
            dict: 所有方法的结果
            
        实现步骤:
        1. 调用所有四种求解方法
        2. 记录计算时间和稳定性参数
        3. 返回比较结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 打印求解信息
        # TODO: 调用四种求解方法
        #   - solve_explicit
        #   - solve_implicit
        #   - solve_crank_nicolson
        #   - solve_with_solve_ivp
        # TODO: 打印每种方法的计算时间和稳定性参数
        # TODO: 返回所有结果的字典
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
            
        实现步骤:
        1. 创建2x2子图
        2. 为每种方法绘制不同时间的解
        3. 设置图例、标签和标题
        4. 可选保存图像
        """
        # TODO: 创建 2x2 子图
        # TODO: 为每种方法绘制解曲线
        # TODO: 设置标题、标签、图例
        # TODO: 可选保存图像
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        
        参数:
            methods_results (dict): compare_methods的结果
            reference_method (str): 参考方法
            
        返回:
            dict: 精度分析结果
            
        实现步骤:
        1. 选择参考解
        2. 计算其他方法与参考解的误差
        3. 统计最大误差和平均误差
        4. 返回分析结果
        """
        # TODO: 验证参考方法存在
        # TODO: 计算各方法与参考解的误差
        # TODO: 统计误差指标
        # TODO: 打印精度分析结果
        # TODO: 返回精度分析字典
        raise NotImplementedError(f"请在 {__file__} 中实现此函数")


def main():
    """
    HeatEquationSolver类的演示。
    """
    # TODO: 创建求解器实例
    # TODO: 比较所有方法
    # TODO: 绘制比较图
    # TODO: 分析精度
    # TODO: 返回结果
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")


if __name__ == "__main__":
    solver, results, accuracy = main()
