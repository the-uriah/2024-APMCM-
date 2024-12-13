import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 房间和设备参数
ROOM_LENGTH = 8.0  # 房间长度 (米)
ROOM_WIDTH = 5.0   # 房间宽度 (米)
ROOM_HEIGHT = 3.0  # 房间高度 (米)
DEVICE_HEIGHT = 0.8  # 总高度 (m)
DEVICE_DIAMETER = 0.35  # 直径 (m)
MAX_POWER = 5000  # 最大功率限制 (W)
TARGET_EFFICIENCY_REDUCTION = 0.85  # 综合能效降低15%

# 网格设置
NX = 30  # x方向网格点数
NY = 30  # y方向网格点数
NZ = 20  # z方向网格点数
DX = ROOM_LENGTH / (NX - 1)
DY = ROOM_WIDTH / (NY - 1)
DZ = ROOM_HEIGHT / (NZ - 1)

# 定义空间坐标
x = np.linspace(0, ROOM_LENGTH, NX)
y = np.linspace(0, ROOM_WIDTH, NY)
z = np.linspace(0, ROOM_HEIGHT, NZ)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# NSGA-II参数设置
POP_SIZE = 100  # 种群大小
MAX_GENERATIONS = 150  # 最大代数
NUM_VARIABLES = 3  # 空调形状比例、过滤面积、加湿面积

# 定义目标函数
def energy_consumption(x):
    purifier_area = x[0] * DEVICE_DIAMETER**2  # 净化器过滤面积
    ac_height = x[1] * DEVICE_HEIGHT          # 空调高度
    humidifier_area = x[2] * DEVICE_DIAMETER**2  # 加湿器湿度影响面积
    base_power = 1000 * ac_height + 5 * abs(purifier_area - 0.06) + 10 * abs(humidifier_area - 0.03)
    # 降低15%的能耗
    return base_power * TARGET_EFFICIENCY_REDUCTION if base_power <= MAX_POWER else np.inf

def air_quality(x):
    purifier_area = x[0] * DEVICE_DIAMETER**2
    humidifier_area = x[2] * DEVICE_DIAMETER**2
    return -(0.8 * purifier_area - 0.1 * humidifier_area)

def humidity_effect(x):
    humidifier_area = x[2] * DEVICE_DIAMETER**2
    return -(0.9 * humidifier_area + 0.1 / (1 + abs(x[1] - 0.5)))  # 空调对湿度的副作用

def objective_function(x):
    energy = energy_consumption(x)
    if energy == np.inf:  # 超出功率限制直接返回不可行解
        return np.array([np.inf, np.inf, np.inf])
    return np.array([
        energy,
        air_quality(x),
        humidity_effect(x)
    ])

# NSGA-II辅助函数
def non_dominated_sorting(fitness_values):
    num_individuals = len(fitness_values)
    ranks = np.zeros(num_individuals)
    for i in range(num_individuals):
        for j in range(num_individuals):
            if np.all(fitness_values[i] <= fitness_values[j]) and not np.all(fitness_values[i] == fitness_values[j]):
                ranks[j] += 1
    return ranks

def calculate_crowding_distance(fitness_values, ranks):
    num_individuals = len(fitness_values)
    crowding_distances = np.zeros(num_individuals)
    for rank in np.unique(ranks):
        indices = np.where(ranks == rank)[0]
        if len(indices) <= 1:
            crowding_distances[indices] = np.inf
            continue
        for m in range(fitness_values.shape[1]):
            sorted_indices = indices[np.argsort(fitness_values[indices, m])]
            crowding_distances[sorted_indices[0]] = np.inf
            crowding_distances[sorted_indices[-1]] = np.inf
            for i in range(1, len(sorted_indices) - 1):
                diff = fitness_values[sorted_indices[i + 1], m] - fitness_values[sorted_indices[i - 1], m]
                max_min_diff = np.ptp(fitness_values[:, m])
                crowding_distances[sorted_indices[i]] += diff / (max_min_diff or 1.0)
    return crowding_distances

def select_parents(population, ranks, crowding_distances):
    selected = []
    for _ in range(POP_SIZE):
        indices = np.where(ranks == np.min(ranks))[0]
        if len(indices) == 1:
            selected.append(population[indices[0]])
            continue
        crowding_distances_subset = crowding_distances[indices]
        selected.append(population[indices[np.argmax(crowding_distances_subset)]])
        ranks[indices[np.argmax(crowding_distances_subset)]] = np.inf
    return np.array(selected)

def generate_offspring(parents):
    offspring = []
    for _ in range(POP_SIZE):
        parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
        child = (parent1 + parent2) / 2 + np.random.normal(0, 0.1, NUM_VARIABLES)
        child = np.clip(child, 0, 1)  # 保证变量范围合法
        offspring.append(child)
    return np.array(offspring)

# NSGA-II主循环
population = np.random.rand(POP_SIZE, NUM_VARIABLES)
for generation in range(MAX_GENERATIONS):
    fitness_values = np.array([objective_function(ind) for ind in population])
    ranks = non_dominated_sorting(fitness_values)
    crowding_distances = calculate_crowding_distance(fitness_values, ranks)
    parents = select_parents(population, ranks, crowding_distances)
    offspring = generate_offspring(parents)
    combined_population = np.vstack((population, offspring))
    fitness_combined = np.array([objective_function(ind) for ind in combined_population])
    population = select_parents(combined_population, non_dominated_sorting(fitness_combined), calculate_crowding_distance(fitness_combined, ranks))

    # 输出当前代的最佳解
    best_index = np.argmin(ranks)
    print(f"Generation {generation + 1}, Best Fitness: {fitness_combined[best_index]}")

# 输出最终结果
optimal_solution = population[np.argmin(ranks)]
print("Optimal Design Parameters:")
print(f"  Air Purifier Filter Area Ratio: {optimal_solution[0]:.3f}")
print(f"  Air Conditioner Height Ratio: {optimal_solution[1]:.3f}")
print(f"  Humidifier Water Area Ratio: {optimal_solution[2]:.3f}")

# 模拟优化后的温度分布
def simulate_combined_system(optimal_solution):
    T = np.full((NX, NY, NZ), 25.0)  # 初始温度
    T[:, :, :] -= optimal_solution[1] * 5  # 空调降温
    return T

# 可视化温度分布
def plot_temperature(T, title):
    mid_z = NZ // 2
    plt.contourf(X[:, :, mid_z], Y[:, :, mid_z], T[:, :, mid_z], levels=25, cmap=cm.jet)
    plt.colorbar(label='Temperature (℃)')
    plt.title(title, fontsize=8)  # 修改标题字体大小
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.show()

# 绘制优化结果的温度分布
T_best = simulate_combined_system(optimal_solution)
plot_temperature(T_best, "Optimized Temperature Distribution")

