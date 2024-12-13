import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Room dimensions and air conditioning parameters
room_width = 5  # Room width (meters)
room_length = 8  # Room length (meters)
room_height = 3  # Room height (meters)
T_outdoor = 35  # Outdoor temperature (Celsius)
T_ac_out = 16  # Air conditioning outlet temperature (Celsius)
T_target = 22  # Target temperature (Celsius)

ac_flow_rate = 600 / 3600  # Air conditioning flow rate (m³/s)

ac_power = 1800  # Air conditioning power (W)
thermal_diffusivity = 0.00002  # Thermal diffusivity (m²/s)

nx, ny, nz = 50, 50, 50  # Grid size corresponding to x, y, and z directions
x = np.linspace(0, room_width, nx)
y = np.linspace(0, room_length, ny)
z = np.linspace(0, room_height, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Air conditioning influence area
def ac_influence(x, y, z, ac_pos, radius=1.5):
    dist = np.sqrt((x - ac_pos[0]) ** 2 + (y - ac_pos[1]) ** 2 + (z - ac_pos[2]) ** 2)
    return np.exp(-dist ** 2 / (2 * radius ** 2))

# Iterate to simulate temperature distribution
def simulate_temperature(ac_pos):
    T = np.full((nx, ny, nz), T_outdoor, dtype=np.float64)
    for t in range(100):  # 100 time steps
        laplacian = (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
                     np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) +
                     np.roll(T, 1, axis=2) + np.roll(T, -1, axis=2) - 6 * T)
        T += laplacian * 0.1  # Time step size
        influence = ac_influence(X, Y, Z, ac_pos)
        T -= influence * (T - T_ac_out) * 0.1

        # Boundary conditions (walls as adiabatic boundary)
        T[0, :, :] = T[:, 0, :] = T[:, -1, :] = T[:, :, 0] = T[:, :, -1] = T_outdoor
    return T


best_position=[2.5, 4.0, 1.5]
# Visualize the effect of the best position
T_best = simulate_temperature(best_position)

# Visualization of the first figure: cross-section temperature distribution
plt.figure(figsize=(6, 6))
cross_section_best = T_best[:, ny // 2, :]  # Cross-section at middle height
im = plt.contourf(x, y, cross_section_best, levels=20, cmap="plasma")
plt.colorbar(im)
plt.xlabel("Width (m)")
plt.ylabel("Length (m)")
plt.title("Temperature Distribution (Cross-section at Best AC Position)")
plt.show()

# Visualization of air conditioning influence weight distribution
plt.figure(figsize=(6, 6))
ax1 = plt.axes(projection='3d')
ac_influence_plot_best = ac_influence(X, Y, Z, best_position)
ax1.scatter(X, Y, Z, c=ac_influence_plot_best.ravel(), cmap="plasma", alpha=0.5, s=5)
ax1.set_title("Air Conditioning Influence Weight Distribution")
ax1.set_xlabel("Width (m)")
ax1.set_ylabel("Length (m)")
ax1.set_zlabel("Height (m)")
plt.show()

# Visualization of the third figure: air conditioning influence weight cross-section
plt.figure(figsize=(6, 6))
ax3 = plt.axes(projection='3d')
ac_influence_plot_slice = ac_influence(X, Y, Z, best_position)[:, :, nz // 2]
X2D, Y2D = np.meshgrid(x, y, indexing='ij')
surf = ax3.plot_surface(X2D, Y2D, ac_influence_plot_slice, cmap="plasma")
plt.colorbar(surf, ax=ax3, shrink=0.5, aspect=10)
ax3.set_title("Air Conditioning Influence Weight Distribution (Cross-section)")
ax3.set_xlabel("Width (m)")
ax3.set_ylabel("Length (m)")
ax3.set_zlabel("Influence Weight")
plt.show()

# Visualization of first figure: cross-section temperature distribution (winter)
plt.figure(figsize=(6, 6))
cross_section = T_best[:, ny // 2, :]  # Cross-section at middle height
im = plt.contourf(x, y, cross_section, levels=20, cmap="viridis")
plt.colorbar(im)
plt.xlabel("Width (m)")
plt.ylabel("Length (m)")
plt.title("Winter Temperature Distribution (Middle Cross-section)")
plt.show()

# Visualization of second figure: air conditioning influence visualization
plt.figure(figsize=(6, 6))
ax2 = plt.axes(projection='3d')
ac_influence_plot = ac_influence(X, Y, Z, best_position)
ax2.scatter(X, Y, Z, c=ac_influence_plot.ravel(), cmap="viridis", alpha=0.5, s=5)
ax2.set_title("Winter Air Conditioning Influence Weight Distribution")
ax2.set_xlabel("Width (m)")
ax2.set_ylabel("Length (m)")
ax2.set_zlabel("Height (m)")
plt.show()

# Visualization of third figure: air conditioning influence weight cross-section
plt.figure(figsize=(6, 6))
ax3 = plt.axes(projection='3d')
ac_influence_plot_slice = ac_influence(X, Y, Z, best_position)[:, :, nz // 2]
X2D, Y2D = np.meshgrid(x, y, indexing='ij')
surf = ax3.plot_surface(X2D, Y2D, ac_influence_plot_slice, cmap="viridis")
plt.colorbar(surf, ax=ax3, shrink=0.5, aspect=10)
ax3.set_title("Winter Air Conditioning Influence Weight Distribution (Cross-section)")
ax3.set_xlabel("Width (m)")
ax3.set_ylabel("Length (m)")
ax3.set_zlabel("Influence Weight")
plt.show()
