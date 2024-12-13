import numpy as np
import matplotlib.pyplot as plt

# 粒子群优化 (PSO) 参数设置
num_particles = 50  # 粒子数量
num_dimensions = 2  # 维度：形状参数和出风角度
max_iterations = 100  # 最大迭代次数
w = 0.5  # 惯性权重
c1 = 1.5  # 个体学习因子
c2 = 1.5  # 社会学习因子


# 定义目标函数
# 目标是最小化能耗并提高室内温度均匀性
# x[0]: 空调的形状比例参数 (0.5 <= x[0] <= 1.5)
# x[1]: 出风角度 (0 <= x[1] <= 90 度)
def objective_function(x):
    shape_factor = x[0]
    angle = x[1]

    # 模拟能耗和温度分布均匀性
    energy_consumption = 1000 * shape_factor + 5 * abs(angle - 45)  # 能耗函数
    temperature_uniformity = 100 / (1 + abs(angle - 30) + shape_factor)  # 温度均匀性

    # 最小化能耗，最大化温度均匀性 (负数用于最大化均匀性)
    return energy_consumption - temperature_uniformity


# 初始化粒子位置和速度
positions = np.random.rand(num_particles, num_dimensions)
positions[:, 0] = 0.5 + positions[:, 0]  # 形状比例初始化范围 [0.5, 1.5]
positions[:, 1] = positions[:, 1] * 90  # 出风角度初始化范围 [0, 90]

velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))

# 初始化个人最优和全局最优
personal_best_positions = np.copy(positions)
personal_best_scores = np.array([objective_function(p) for p in positions])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = min(personal_best_scores)

# 迭代优化
for iteration in range(max_iterations):
    for i in range(num_particles):
        # 更新速度和位置
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (personal_best_positions[i] - positions[i]) +
                         c2 * r2 * (global_best_position - positions[i]))
        positions[i] += velocities[i]

        # 确保粒子在范围内
        positions[i, 0] = np.clip(positions[i, 0], 0.5, 1.5)
        positions[i, 1] = np.clip(positions[i, 1], 0, 90)

        # 更新个人最优
        score = objective_function(positions[i])
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]

        # 更新全局最优
        if score < global_best_score:
            global_best_score = score
            global_best_position = positions[i]

    # 打印每次迭代的最佳得分
    print(f"第 {iteration + 1} 次迭代, 最佳得分: {global_best_score}")

# 打印优化后的结果
print("最佳形状比例参数:", global_best_position[0])
print("最佳出风角度:", global_best_position[1])

