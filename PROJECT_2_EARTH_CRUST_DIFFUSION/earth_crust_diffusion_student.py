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
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

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
