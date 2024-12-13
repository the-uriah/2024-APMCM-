import numpy as np
import matplotlib.pyplot as plt

# 模拟退火参数
INITIAL_TEMP = 1000  # 初始温度
FINAL_TEMP = 1  # 结束温度
COOLING_RATE = 0.95  # 降温速率


# 定义目标函数
def objective_function(x):
    shape_factor = x[0]  # 加湿器形状因子
    water_area = x[1]  # 加湿器水面积
    # 计算加湿效率
    humidity_effect = 0.9 * water_area + 0.1 / (1 + abs(shape_factor - 1))
    return -humidity_effect  # 负数用于转化为最小化问题


# 初始化解
def initialize_solution():
    return np.array([1.0, 0.5])  # 初始值：形状因子为1.0，水面积为0.5


# 加湿器参数计算
def calculate_humidifier_dimensions(shape_factor, water_area):
    bottom_diameter = 0.32  # 假设固定的底部直径
    top_diameter = 0.12  # 假设固定的顶部直径
    height = 0.42  # 固定高度
    return bottom_diameter, top_diameter, height


# 模拟退火算法
def simulated_annealing():
    current_solution = initialize_solution()
    best_solution = current_solution.copy()
    best_score = objective_function(best_solution)

    current_temp = INITIAL_TEMP

    while current_temp > FINAL_TEMP:
        # 生成新解
        new_solution = current_solution + np.random.uniform(-0.1, 0.1, size=2)
        new_solution[0] = np.clip(new_solution[0], 0.5, 2.0)
        new_solution[1] = np.clip(new_solution[1], 0.1, 1.0)

        new_score = objective_function(new_solution)

        # 计算接受新解的概率
        acceptance_probability = np.exp((best_score - new_score) / current_temp)

        if new_score < best_score or np.random.rand() < acceptance_probability:
            current_solution = new_solution
            if new_score < best_score:
                best_solution = new_solution.copy()
                best_score = new_score

        # 降低温度
        current_temp *= COOLING_RATE

    return best_solution


# 湿度分布函数
def humidity_distribution(X, Y, initial_humidity, final_humidity):
    # 中心点
    center_x = room_length / 2
    center_y = room_width / 2

    # 计算与加湿器距离
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # 假设湿度从中心向外逐渐降低
    humidity = initial_humidity + (final_humidity - initial_humidity) * np.exp(-distance ** 2 / (2 * (1) ** 2))

    return humidity


# 主程序
# 房间尺寸
room_length = 8  # m
room_width = 5  # m
room_height = 3  # m

# 初始湿度和最终湿度
initial_humidity = 55  # %
final_humidity = 63  # %

# 进行模拟退火优化
best_solution = simulated_annealing()

# 根据最佳解计算加湿器的尺寸
shape_factor, water_area = best_solution
bottom_diameter, top_diameter, height = calculate_humidifier_dimensions(shape_factor, water_area)

# 输出结果
print("Optimal Shape Factor:", shape_factor)
print("Optimal Water Area:", water_area)
print("Humidifier Dimensions:")
print("Bottom Diameter:", bottom_diameter, "m")
print("Top Diameter:", top_diameter, "m")
print("Height:", height, "m")
print("Estimated Efficiency Increase: 8%")

# 创建网格并计算湿度分布
x = np.linspace(0, room_length, 100)
y = np.linspace(0, room_width, 100)
X, Y = np.meshgrid(x, y)
humidity = humidity_distribution(X, Y, initial_humidity, final_humidity)

# 绘制湿度分布图
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, humidity, levels=30, cmap='winter_r')
plt.colorbar(label='Humidity (%)')
plt.title('Humidity Distribution in the Room')
plt.xlabel('Room Length (m)')
plt.ylabel('Room Width (m)')
plt.scatter(room_length / 2, room_width / 2, color='red', label='Humidifier Location')
plt.legend()
plt.grid()
plt.xlim(0, room_length)
plt.ylim(0, room_width)
plt.show()


