import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Set up support for Chinese fonts
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Define room and physical parameters

# Room dimensions (meters)
room_length = 8.0  # Length
room_width = 5.0   # Width
room_height = 3.0  # Height

# Grid division
nx = 30  # Number of grid points in the x direction, reduced to improve computational efficiency
ny = 30  # Number of grid points in the y direction
nz = 20  # Number of grid points in the z direction
dx = room_length / (nx - 1)  # Grid spacing (x direction)
dy = room_width / (ny - 1)   # Grid spacing (y direction)
dz = room_height / (nz - 1)  # Grid spacing (z direction)

x = np.linspace(0, room_length, nx)  # x coordinates
y = np.linspace(0, room_width, ny)   # y coordinates
z = np.linspace(0, room_height, nz)  # z coordinates

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Ensure consistent indexing order

# Time step and total time
dt = 0.05  # seconds, decrease time step to improve stability
total_time = 1000  # total time
nt = int(total_time / dt)  # number of time steps

# Thermal diffusivity (air)
alpha = 0.0002  # thermal diffusivity, units m^2/s

# Initial temperature field (summer and winter)
initial_temp_summer = 30.0  # initial temperature in summer
initial_temp_winter = 10.0  # initial temperature in winter
target_temp = 22.0           # target temperature

# External environment temperature
outside_temp_summer = 35.0  # summer outside temperature
outside_temp_winter = 5.0   # winter outside temperature

# Air conditioning parameters
T_outdoor = 35  # outdoor temperature (Celsius)
T_ac_out = 16   # air conditioning outlet temperature (Celsius)
T_target = 22   # target temperature (Celsius)

ac_position = (2.5, 4.0, 1.5)  # initial position of the air conditioner (x,y,z)

# 2. Define different shapes and sizes for the air conditioner

def get_ac_mask(shape, size, position):
    """
    Generate a mask for the air conditioning area based on its shape, size, and position.
    """
    mask = np.zeros((nx, ny, nz), dtype=bool)
    if shape == 'rectangle':
        # size = (width, height, depth)
        width, height, depth = size
        x_center, y_center, z_center = position

        # Calculate start and end indices
        x_start = int((x_center - width / 2) / dx)
        y_start = int((y_center - height / 2) / dy)
        z_start = int((z_center - depth / 2) / dz)
        x_end = int((x_center + width / 2) / dx)
        y_end = int((y_center + height / 2) / dy)
        z_end = int((z_center + depth / 2) / dz)

        # Ensure indices are within valid range
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        z_start = max(0, z_start)
        x_end = min(nx, x_end)
        y_end = min(ny, y_end)
        z_end = min(nz, z_end)

        mask[x_start:x_end, y_start:y_end, z_start:z_end] = True

    elif shape == 'circle':
        # size = radius
        radius = size
        cx = int(position[0] / dx)
        cy = int(position[1] / dy)
        cz = int(position[2] / dz)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    distance = ((i - cx) * dx) ** 2 + ((j - cy) * dy) ** 2 + ((k - cz) * dz) ** 2
                    if distance <= radius ** 2:
                        mask[i, j, k] = True
    return mask

# 3. Establish the numerical simulation model for the temperature field

def simulate_temperature(ac_shape, ac_size, initial_temp, outside_temp):
    """
    Simulate the temperature changes given the air conditioner's shape and size.
    """
    T = np.full((nx, ny, nz), initial_temp)  # Initialize temperature field
    T_new = T.copy()  # Temporary variable to update temperature field

    # Get the air conditioning region mask
    ac_mask = get_ac_mask(ac_shape, ac_size, position=(room_length / 2, room_width / 2, room_height / 2))

    # Simulation process
    for n in range(nt):
        # Heat conduction calculation (3D heat conduction equation)
        T_new[1:-1, 1:-1, 1:-1] = T[1:-1, 1:-1, 1:-1] + alpha * dt * (
            (T[2:, 1:-1, 1:-1] - 2 * T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / dx ** 2 +
            (T[1:-1, 2:, 1:-1] - 2 * T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / dy ** 2 +
            (T[1:-1, 1:-1, 2:] - 2 * T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / dz ** 2
        )

        # Boundary conditions (surrounding walls)
        T_new[0, :, :] = outside_temp
        T_new[-1, :, :] = outside_temp
        T_new[:, 0, :] = outside_temp
        T_new[:, -1, :] = outside_temp
        T_new[:, :, 0] = outside_temp
        T_new[:, :, -1] = outside_temp

        # Set air conditioning area temperature to target temperature
        T_new[ac_mask] = target_temp

        # Update temperature field
        T = T_new.copy()

    return T

# Function to evaluate the air conditioner position
def evaluate_ac_position(ac_pos, ac_size, ac_shape):
    T = simulate_temperature(ac_shape, ac_size, initial_temp_summer, T_outdoor)
    target_temp_arr = np.full((nx, ny, nz), T_target, dtype=np.float64)
    mse = np.mean((T - target_temp_arr) ** 2)  # Mean squared error
    return mse

# Grid search to find the best position
def find_best_ac_position():
    x_positions = np.linspace(0.5, room_width - 0.5, 10)  # Within room width range
    y_positions = np.linspace(0.5, room_length - 0.5, 10)  # Within room length range
    z_positions = np.linspace(0.5, room_height - 0.5, 5)   # Within room height range

    best_position = ac_position  # Initial position
    best_error = evaluate_ac_position(best_position)  # Initial error

    # Grid search to find the best position
    for x in x_positions:
        for y in y_positions:
            for z in z_positions:
                current_position = (x, y, z)
                current_error = evaluate_ac_position(current_position)
                if current_error < best_error:
                    best_error = current_error
                    best_position = current_position

    return best_position, best_error

# 4. Simulate temperature changes in summer and winter

# Define the list of shapes and sizes for the air conditioner
ac_shapes = ['rectangle', 'circle']
ac_sizes = {
    'rectangle': [(1, 0.5, 2), (1.0, 0.5, 0.5), (1.5, 0.7, 0.7), (1.8, 0.6, 0.4)],  # Different sizes for rectangular AC (width, height, depth)
    'circle': [0.35, 0.4, 0.45, 0.5, 0.55, 0.65]  # Different radius sizes for circular AC
}

# Store results
results = []

for shape in ac_shapes:
    sizes = ac_sizes[shape]
    for size in sizes:
        # Simulate summer
        T_summer = simulate_temperature(shape, size, initial_temp_summer, outside_temp_summer)
        # Calculate summer temperature variance
        variance_summer = np.var(T_summer - target_temp)

        # Simulate winter
        T_winter = simulate_temperature(shape, size, initial_temp_winter, outside_temp_winter)
        # Calculate winter temperature variance
        variance_winter = np.var(T_winter - target_temp)

        # Total variance
        total_variance = variance_summer + variance_winter

        # Save results
        results.append({
            'shape': shape,
            'size': size,
            'variance_summer': variance_summer,
            'variance_winter': variance_winter,
            'total_variance': total_variance
        })

        print(f"Shape: {shape}, Size: {size}, Total Temperature Variance: {total_variance:.2f}")

# 5. Compare the performance of different air conditioning designs and choose the best solution

# Find the design with the minimum total temperature variance
best_result = min(results, key=lambda x: x['total_variance'])

print("\nBest Air Conditioning Design:")
print(f"Shape: {best_result['shape']}")
print(f"Size: {best_result['size']}")
print(f"Summer Temperature Variance: {best_result['variance_summer']:.2f}")
print(f"Winter Temperature Variance: {best_result['variance_winter']:.2f}")
print(f"Total Temperature Variance: {best_result['total_variance']:.2f}")

# 6. Visualize the temperature distribution of the best scheme

def plot_temperature(T, title):
    """
    Plot the 2D cross-section of the temperature distribution in the middle layer.
    """
    plt.figure(figsize=(8, 6))
    mid_z = nz // 2
    plt.contourf(X[:, :, mid_z], Y[:, :, mid_z], T[:, :, mid_z], 20, cmap=cm.jet)
    plt.colorbar(label='Temperature (℃)')
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')

    # Draw the air conditioning area boundary
    ac_shape = best_result['shape']
    ac_size = best_result['size']
    ac_mask = get_ac_mask(ac_shape, ac_size, position=(room_length / 2, room_width / 2, room_height / 2))
    ac_slice = ac_mask[:, :, mid_z]
    plt.contour(X[:, :, mid_z], Y[:, :, mid_z], ac_slice.astype(int), levels=[0.5], colors='blue', linewidths=2)

    plt.show()

# Simulate and plot the temperature distribution for the best scheme (summer)
T_summer_best = simulate_temperature(best_result['shape'], best_result['size'], initial_temp_summer,
                                     outside_temp_summer)
plot_temperature(T_summer_best, 'Summer Best Air Conditioning Design Temperature Distribution')

# Simulate and plot the temperature distribution for the best scheme (winter)
T_winter_best = simulate_temperature(best_result['shape'], best_result['size'], initial_temp_winter,
                                     outside_temp_winter)
plot_temperature(T_winter_best, 'Winter Best Air Conditioning Design Temperature Distribution')

# Find and return the best air conditioning position
best_position, best_error = find_best_ac_position()
best_position = tuple(float(coord) for coord in best_position)  # Convert to regular numbers
print(f"Best Air Conditioner Position: {best_position}, Mean Squared Error: {best_error:.2f}")
