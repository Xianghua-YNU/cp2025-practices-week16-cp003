"""
学生模板：地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    
    物理背景: 模拟地壳中温度随深度和时间的周期性变化
    数值方法: 显式差分格式
    
    实现步骤:
    1. 设置物理参数和网格参数
    2. 初始化温度场
    3. 应用边界条件
    4. 实现显式差分格式
    5. 返回计算结果
    """
    # TODO: 设置物理参数
    # TODO: 初始化数组
    # TODO: 实现显式差分格式
    # TODO: 返回计算结果
        # 1. 设置物理参数
    D = 0.1  # 热扩散率 (m²/day)
    A = 10   # 年平均地表温度 (°C)
    B = 12   # 地表温度振幅 (°C)
    tau = 365  # 年周期 (days)
    total_depth = 20  # 模拟深度 (m)
    total_time = 3 * 365  # 模拟总时间 (3年)
    
    # 2. 设置网格参数 (需要满足稳定性条件)
    dz = 0.1  # 空间步长 (m)
    dt = 0.01  # 时间步长 (days)
    
    # 计算网格点数
    n_z = int(total_depth / dz) + 1
    n_t = int(total_time / dt) + 1
    
    # 稳定性检查
    r = D * dt / (dz ** 2)
    if r > 0.5:
        raise ValueError(f"稳定性条件不满足: r = {r:.2f} > 0.5")
    
    # 3. 初始化数组
    depth_array = np.linspace(0, total_depth, n_z)
    time_array = np.linspace(0, total_time, n_t)
    temperature = np.zeros((n_t, n_z))
    
    # 4. 设置初始条件
    temperature[0, :] = 10  # 初始温度 10°C
    temperature[0, 0] = A + B * np.sin(2 * np.pi * 0 / tau)  # 地表边界
    temperature[0, -1] = 11  # 深层边界
    
    # 5. 时间推进求解
    for k in range(0, n_t - 1):
        # 应用边界条件
        current_time = time_array[k]
        temperature[k+1, 0] = A + B * np.sin(2 * np.pi * current_time / tau)  # 地表边界
        temperature[k+1, -1] = 11  # 深层边界
        
        # 显式差分格式
        for i in range(1, n_z - 1):
            temperature[k+1, i] = temperature[k, i] + r * (
                temperature[k, i+1] - 2 * temperature[k, i] + temperature[k, i-1]
            )
    
    return depth_array, temperature

if __name__ == "__main__":
    # 测试代码
    try:
        depth, T = solve_earth_crust_diffusion()
        print(f"计算完成，温度场形状: {T.shape}")
        # 绘制结果
        plt.figure(figsize=(10, 6))
        
        # 选择四个季节的时间点 (1年=365天)
        seasons = {
            'Spring': 365 + 80,   # 春季 (第1年3月21日左右)
            'Summer': 365 + 172,   # 夏季 (第1年6月21日左右)
            'Autumn': 365 + 265,   # 秋季 (第1年9月22日左右)
            'Winter': 365 + 355    # 冬季 (第1年12月21日左右)
        }
        
        for season, day in seasons.items():
            time_index = int(day / 0.01)  # 转换为时间索引
            plt.plot(T[time_index, :], depth, label=f'{season}')
        
        plt.gca().invert_yaxis()  # 深度向下为正
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Depth (m)')
        plt.title('Seasonal Temperature Profiles in Earth Crust')
        plt.legend()
        plt.grid(True)
        plt.show()
    except NotImplementedError as e:
        print(e)
