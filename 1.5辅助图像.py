import matplotlib.pyplot as plt
import numpy as np
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']  # Support for Chinese characters
plt.rcParams['axes.unicode_minus'] = False

# Ignore warnings
warnings.filterwarnings('ignore')

# Parameter settings
num_particles = 175  # Number of particles
num_dimensions = 3   # Number of dimensions (Height H, Diameter D, Exit Angle θ)
max_iter = 600      # Maximum number of iterations

bounds = np.array([
    [0.6, 0.8],     # Height H range (meters)
    [0.3, 0.5],     # Diameter D range (meters)
    [30, 60]        # Exit angle θ range (degrees)
])

class Particle:
    def __init__(self, num_particles, num_dimensions, bounds, max_iter, w, c1, c2):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.w = w              # Inertia weight
        self.c1 = c1            # Cognitive coefficient
        self.c2 = c2            # Social coefficient

        # Initialize particle positions and velocities
        self.positions = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.num_particles, self.num_dimensions))
        self.velocities = np.zeros((self.num_particles, self.num_dimensions))

        # Initialize personal best and global best
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(self.num_particles, np.inf)
        self.gbest_score = np.inf
        self.gbest_position = None

    def simulate_CFD(self, H, D, theta):
        """
        Simulate TUI and E using room dynamics, airflow parameters, and air conditioning efficiency.
        """
        room_volume = 5 * 8 * 3  # Room volume (m³)
        ac_flow_rate = 600 / 3600  # Airflow rate (m³/s)
        delta_T = 12.0  # Temperature difference by AC (°C)
        ac_power = 1800  # Air conditioner power (W)

        # CFD-Based Temperature Distribution
        num_points = 1000  # Simulate 1000 points in the room
        x_points = np.random.uniform(0, 5, num_points)
        y_points = np.random.uniform(0, 8, num_points)
        z_points = np.random.uniform(0, 3, num_points)

        distances = np.sqrt((x_points - 2.5) ** 2 + (y_points - 4.0) ** 2 + (z_points - 2.0) ** 2)
        cooling_effect = np.exp(-distances / (D * theta / 45)) * delta_T

        T_room = 35 - cooling_effect  # Initial temperature = 35°C
        TUI = np.std(T_room)  # Standard deviation of temperature differences

        efficiency_factor = (0.8 - (H - 0.6) / 0.2) * (1 - TUI / 100)  # Efficiency drops with non-optimal H and high TUI
        E = ac_power * efficiency_factor
        E = min(E, 1800)
        return TUI, E

    def fitness(self, position):
        H, D, theta = position
        TUI, E = self.simulate_CFD(H, D, theta)
        w1, w2 = 0.7, 0.3   # Weight coefficients
        E_max = 1800        # Maximum energy consumption (for normalization)
        r = w1 * TUI + w2 * E / E_max  # Normalization
        return r

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_particles):
                r_i = self.fitness(self.positions[i])
                if r_i < self.pbest_scores[i]:
                    self.pbest_scores[i] = r_i
                    self.pbest_positions[i] = self.positions[i].copy()
                if r_i < self.gbest_score:
                    self.gbest_score = r_i
                    self.gbest_position = self.positions[i].copy()

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.gbest_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])

            print(f"Iteration {iter + 1}/{self.max_iter}, Best fitness: {self.gbest_score:.4f}")

        print("Optimal design parameters:")
        print(f"Height H = {self.gbest_position[0]:.3f} m")
        print(f"Diameter D = {self.gbest_position[1]:.3f} m")
        print(f"Exit Angle θ = {self.gbest_position[2]:.1f} degrees")

pso = Particle(num_particles, num_dimensions, bounds, max_iter, w=0.5, c1=2.0, c2=2.0)
pso.optimize()

# Summer temperature simulation
# ------------------------------
t_total = 4000  # Total simulation time (seconds)
dt = 0.05       # Time step (seconds)
time = np.arange(0, t_total + dt, dt)

rho = 1.2
c_p = 1005
room_width = 5  # Room width (meters)
room_length = 8  # Room length (meters)
room_height = 3  # Room height (meters)
T_initial_summer = 35  # Initial indoor temperature (°C)
T_outdoor_summer = 35  # Outdoor temperature (°C)
T_ac_out_summer = 16   # Air conditioning outlet temperature (°C)
T_target_summer = 26   # Target temperature (°C)
delta_T_ac_summer = 12.0  # Temperature difference of AC (°C), cooling mode
ac_flow_rate_summer = 600 / 3600  # Air conditioning airflow (m³/s)
ac_position = (2.5, 4.0, 2.0)  # Air conditioning position (x, y, z)

nx, ny, nz = 50, 50, 50
x = np.linspace(0, room_width, nx)
y = np.linspace(0, room_length, ny)
z = np.linspace(0, room_height, nz)
X, Y, Z = np.meshgrid(x, y, z)

T = np.full((nx, ny, nz), T_outdoor_summer, dtype=np.float64)

def ac_influence(x, y, z, ac_pos, radius=1.5):
    dist = np.sqrt((x - ac_pos[0]) ** 2 + (y - ac_pos[1]) ** 2 + (z - ac_pos[2]) ** 2)
    return np.exp(-dist ** 2 / (2 * radius ** 2))

timesteps = 400
for t in range(timesteps):
    laplacian = (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
                 np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) +
                 np.roll(T, 1, axis=2) + np.roll(T, -1, axis=2) - 6 * T)
    T += laplacian * dt
    influence = ac_influence(X, Y, Z, ac_position)
    T -= influence * (T - T_ac_out_summer) * dt  # Cooling mode

    T[0, :, :] = T[:, 0, :] = T[:, -1, :] = T[:, :, 0] = T[:, :, -1] = T_outdoor_summer

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
cross_section = T[:, ny // 2, :]
im = ax1.contourf(x, y, cross_section, levels=20, cmap="plasma")
plt.colorbar(im, ax=ax1)
ax1.set_xlabel("Width (m)")
ax1.set_ylabel("Length (m)")
ax1.set_title("Temperature Distribution (Middle Cross Section)")

ax2 = fig.add_subplot(122, projection='3d')
ac_influence_plot = ac_influence(X, Y, Z, ac_position)
ax2.scatter(X, Y, Z, c=ac_influence_plot.ravel(), cmap="plasma", alpha=0.3, s=2)
ax2.set_title("Air Conditioning Influence Weight Distribution")
ax2.set_xlabel("Width (m)")
ax2.set_ylabel("Length (m)")
ax2.set_zlabel("Height (m)")

plt.tight_layout()
plt.show()

T_room_summer = np.zeros(len(time))
T_room_summer[0] = T_initial_summer

V_room = room_length * room_width * room_height
m_air = rho * V_room

alpha = ac_flow_rate_summer / m_air  # Mass flow rate coefficient (1/s)
U = 1.0  # W/(m²·K)
A_wall = 2 * (room_length * room_height + room_width * room_height + room_length * room_width)
beta = U * A_wall / (m_air * c_p)

# Summer temperature change simulation
for i in range(1, len(time)):
    T_current = T_room_summer[i - 1]
    T_ac_out = T_current - delta_T_ac_summer  # Cooling mode

    dTdt = - (alpha + beta) * T_current + alpha * T_ac_out + beta * T_outdoor_summer
    T_next = T_current + dTdt * dt

    T_room_summer[i] = T_next

    if T_next <= T_target_summer:
        print(f"At {time[i]} seconds, the indoor temperature reaches the target temperature of {T_target_summer}°C.")
        T_room_summer = T_room_summer[:i + 1]
        time = time[:i + 1]
        break

plt.figure(figsize=(10, 6))
plt.plot(time / 60, T_room_summer, label="Indoor Temperature")
plt.axhline(y=T_target_summer, color='yellow', linestyle='--', label='Target Temperature (26.0 °C)')
plt.axhline(y=T_initial_summer, color='purple', linestyle='--', label='Initial Temperature (35.0 °C)')

start_time = 0  # Start time
start_temp = T_room_summer[0]  # Start temperature
plt.plot([start_time / 60, start_time / 60], [0, start_temp], color='gray', linestyle='--')
plt.plot([0, start_time / 60], [start_temp, start_temp], color='gray', linestyle='--')
plt.text(start_time / 60, start_temp, f'Start\n({start_time / 60:.2f}, {start_temp:.2f})',
         horizontalalignment='right', verticalalignment='bottom', color='gray')

end_time = time[len(T_room_summer) - 1]  # End time
end_temp = T_room_summer[-1]  # End temperature
plt.plot([end_time / 60, end_time / 60], [0, end_temp], color='gray', linestyle='--')
plt.plot([0, end_time / 60], [end_temp, end_temp], color='gray', linestyle='--')
plt.text(end_time / 60, end_temp, f'End\n({end_time / 60:.2f}, {end_temp:.2f})',
         horizontalalignment='right', verticalalignment='bottom', color='gray')

plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.title('Indoor Temperature Change Over Time (Summer)')
plt.legend()

plt.ylim(20, 40)  # Set y-axis range
plt.gca().set_yticks(np.linspace(20, 40, 21))  # Set y-axis to uniform distribution
plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9)

plt.show()

# Winter simulation
# ---------------------------------------------------
# Time parameters
t_total = 5000  # Total simulation time (seconds)
dt = 0.08       # Time step (seconds)
time = np.arange(0, t_total + dt, dt)

# Define constants and initial parameters
rho = 1.2  # Air density (kg/m³)
c_p = 1005  # Specific heat capacity of air (J/(kg·K))
room_width = 5  # Room width (meters)
room_length = 8  # Room length (meters)
room_height = 3  # Room height (meters)
T_initial = 10  # Initial indoor temperature (°C)
T_outdoor = 5   # Outdoor temperature (°C)
T_ac_out = 25   # Air conditioning outlet temperature (°C)
T_target = 24   # Target temperature (°C)
delta_T_ac = 20.0  # Temperature difference of AC (°C), heating mode
ac_flow_rate = 700 / 3600  # Air conditioning airflow (m³/s)
ac_position = (2.5, 4.0, 2.0)  # Air conditioning position (x, y, z)

# Grid division
nx, ny, nz = 50, 50, 50
x = np.linspace(0, room_width, nx)
y = np.linspace(0, room_length, ny)
z = np.linspace(0, room_height, nz)
X, Y, Z = np.meshgrid(x, y, z)

# Initialize temperature field
T = np.full((nx, ny, nz), T_outdoor, dtype=np.float64)

# Air conditioning influence region
def ac_influence(x, y, z, ac_pos, radius=1.5):
    dist = np.sqrt((x - ac_pos[0]) ** 2 + (y - ac_pos[1]) ** 2 + (z - ac_pos[2]) ** 2)
    return np.exp(-dist ** 2 / (2 * radius ** 2))

# Iteratively simulate temperature distribution
timesteps = 400
for t in range(timesteps):
    laplacian = (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
                 np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) +
                 np.roll(T, 1, axis=2) + np.roll(T, -1, axis=2) - 6 * T)
    T += laplacian * dt
    influence = ac_influence(X, Y, Z, ac_position)
    T += influence * (T_ac_out - T) * dt  # Heating mode

    T[0, :, :] = T[:, 0, :] = T[:, -1, :] = T[:, :, 0] = T[:, :, -1] = T_outdoor

# Visualize results
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
cross_section = T[:, ny // 2, :]
im = ax1.contourf(x, y, cross_section, levels=20, cmap="plasma")
plt.colorbar(im, ax=ax1)
ax1.set_xlabel("Width (m)")
ax1.set_ylabel("Length (m)")
ax1.set_title("Temperature Distribution (Middle Cross Section)")

ax2 = fig.add_subplot(122, projection='3d')
ac_influence_plot = ac_influence(X, Y, Z, ac_position)
ax2.scatter(X, Y, Z, c=ac_influence_plot.ravel(), cmap="plasma", alpha=0.3, s=2)
ax2.set_title("Air Conditioning Influence Weight Distribution")
ax2.set_xlabel("Width (m)")
ax2.set_ylabel("Length (m)")
ax2.set_zlabel("Height (m)")

plt.tight_layout()
plt.show()

# Simulate temperature changes
T_room = np.zeros(len(time))
T_room[0] = T_initial  # Initial indoor temperature

alpha = ac_flow_rate / m_air  # Mass flow rate coefficient (1/s)
beta = U * A_wall / (m_air * c_p)  # Heat transfer coefficient (1/s)

# Simulate temperature changes
for i in range(1, len(time)):
    T_current = T_room[i - 1]
    T_ac_out = T_current + delta_T_ac  # Heating mode

    dTdt = - (alpha + beta) * T_current + alpha * T_ac_out + beta * T_outdoor
    T_next = T_current + dTdt * dt

    T_room[i] = T_next

    if T_next >= T_target:
        print(f"At {time[i]} seconds, the indoor temperature reaches the target temperature of {T_target}°C.")
        T_room = T_room[:i + 1]
        time = time[:i + 1]
        break

plt.figure(figsize=(10, 6))
plt.plot(time / 60, T_room, label="Indoor Temperature")
plt.axhline(y=T_target, color='yellow', linestyle='--', label='Target Temperature (24.0 °C)')
plt.axhline(y=T_initial, color='purple', linestyle='--', label='Initial Temperature (10.0 °C)')

start_time = 0  # Start time
start_temp = T_room[0]  # Start temperature
plt.plot([start_time / 60, start_time / 60], [0, start_temp], color='gray', linestyle='--')
plt.plot([0, start_time / 60], [start_temp, start_temp], color='gray', linestyle='--')
plt.text(start_time / 60, start_temp, f'Start\n({start_time / 60:.2f}, {start_temp:.2f})',
         horizontalalignment='right', verticalalignment='bottom', color='gray')

end_time = time[len(T_room) - 1]  # End time
end_temp = T_room[-1]  # End temperature
plt.plot([end_time / 60, end_time / 60], [0, end_temp], color='gray', linestyle='--')
plt.plot([0, end_time / 60], [end_temp, end_temp], color='gray', linestyle='--')
plt.text(end_time / 60, end_temp, f'End\n({end_time / 60:.2f}, {end_temp:.2f})',
         horizontalalignment='right', verticalalignment='bottom', color='gray')

plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.title('Indoor Temperature Change Over Time (Winter)')
plt.legend()

plt.ylim(0, 30)  # Set y-axis range
plt.gca().set_yticks(np.linspace(0, 30, 31))  # Set y-axis to uniform distribution

plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9)

plt.show()
