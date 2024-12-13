import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from joblib import Parallel, delayed  # Import joblib for parallel processing

# Set parameters
room_length = 8.0  # Room length: 8 meters
room_width = 5.0   # Room width: 5 meters
room_height = 3.0  # Room height: 3 meters

# Increased Grid Division for higher resolution
nx = 17  # Number of grid points in x direction
ny = 17  # Number of grid points in y direction
nz = 9 # Number of grid points in z direction
dx = room_length / (nx - 1)
dy = room_width / (ny - 1)
dz = room_height / (nz - 1)
x = np.linspace(0, room_length, nx)
y = np.linspace(0, room_width, ny)
z = np.linspace(0, room_height, nz)
X, Y, Z = np.meshgrid(x, y, z)

# Airflow and Pollutant Diffusion Coefficient
alpha = 0.1

# Simulate Air Purification Effect
def simulate_purification(filter_area, efficiency, pollution_concentration):
    T = np.copy(pollution_concentration)
    T_new = T.copy()

    # Iterate for the simulation process (reduced iterations for optimization)
    for _ in range(50):  # Reduced from 100 to 50
        T_new = T * (1 - efficiency)  # Apply effectiveness
        T_new[1:-1, 1:-1, 1:-1] += alpha * (
            T[1:-1, 1:-1, 2:] + T[1:-1, 1:-1, :-2] +
            T[1:-1, 2:, 1:-1] + T[1:-1, :-2, 1:-1] +
            T[2:, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1] -
            6 * T[1:-1, 1:-1, 1:-1]
        )
        T = T_new.copy()

    return T

# Objective Function with caching
fitness_cache = {}

def objective(params):
    filter_area = params[0]
    efficiency = params[1] / 100
    pollution_concentration = np.random.uniform(0.5, 1.0, (nz, ny, nx))

    # Caching the fitness value
    cache_key = (filter_area, efficiency)
    if cache_key in fitness_cache:
        return fitness_cache[cache_key]

    T = simulate_purification(filter_area, efficiency, pollution_concentration)

    # Apply penalties if constraints are not met
    penalty = 0
    if efficiency < 0.93:
        penalty += 1 / (0.93 - efficiency)  # Higher penalty for lower efficiency
    if np.abs(filter_area - 0.55) > 0.01:  # Allow a small tolerance
        penalty += 1 / (0.55 - filter_area)

    final_fit = np.var(T) + penalty
    fitness_cache[cache_key] = final_fit  # Cache the result
    return final_fit

# Genetic Algorithm (GA) Parameters
population_size = 100
generations = 500
mutation_rate = 0.1

# Initialize Population
population = np.zeros((population_size, 2))
population[:, 0] = 0.55  # Set filter area directly to 0.55
population[:, 1] = np.random.rand(population_size) * 100  # Random efficiency range [0, 100]

# Genetic Algorithm
for generation in range(generations):
    # Calculate fitness with parallel processing
    fitness = Parallel(n_jobs=-1)(delayed(objective)(individual) for individual in population)

    # Selection
    idx = np.argsort(fitness)[:population_size // 2]  # Select the best half
    selected = population[idx]

    # Crossover
    offspring = []
    for _ in range(population_size - len(selected)):
        parents = selected[np.random.choice(len(selected), 2, replace=False)]
        child = np.copy(parents[0])  # Create child from one parent
        if np.random.rand() < 0.5:
            child[1] = parents[1][1]  # Inherit efficiency from the other parent
        offspring.append(child)

    # Mutation
    for individual in offspring:
        if np.random.rand() < mutation_rate:
            individual[1] += np.random.randn() * 1.0  # Mutate the efficiency
            individual[1] = np.clip(individual[1], 0, 100)  # Ensure it's within bounds

    # Update the population
    population = np.vstack((selected, offspring))

# Find the best solution
best_solution = population[np.argmin([objective(individual) for individual in population])]
best_area = best_solution[0]
best_efficiency = best_solution[1]

# Ensure results meet criteria
if best_efficiency / 100 >= 0.93:
    print(f"Best Filter Area: {best_area:.2f} square meters, Purification Efficiency: {best_efficiency:.2f}%")
else:
    print("Could not find a solution with the specified criteria.")

# Calculate purification effect under the best size
best_purification = simulate_purification(best_area, best_efficiency / 100, np.random.uniform(0.5, 1.0, (nz, ny, nx)))

# Visualize the purification effect
def plot_purification(T, title):
    plt.figure(figsize=(8, 6))  # Increased figure size for better visibility
    plt.contourf(X[:, :, 0], Y[:, :, 0], T[nz // 2, :, :], levels=35, cmap=cm.jet)  # Increase levels for better color gradient
    plt.colorbar(label='Pollutant Concentration')
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.show()

# Visualize the purification effect of the best design
plot_purification(best_purification, 'Purification Effect of the Best Honeycomb Air Purifier Design')
