import numpy as np
import matplotlib.pyplot as plt

# Constants for the simulation
m = 1  # Mass of each particle (kg)
dt = 0.1  # Time step for the simulation (s)
n_particles = 100  # Number of particles
speed_range = 5.0  # Speed range for initial particle velocities
box_size = 10.0  # Size of the simulation box

# Interactions class for collision and velocity change estimation
class Interactions:
    def __init__(self, M, m):
        self.M = M  # Mass of the sphere in terms of particles' masses
        self.m = m  # Mass of each particle

    def collision(self, velocities):
        x_velocities = velocities[:, 0]
        y_velocities = velocities[:, 1]
        Mvx = []
        Mvy = []
        for i in range(len(x_velocities)):
            v1 = x_velocities[i]
            v2 = v1 * (self.m - self.M) / (self.m + self.M)
            v3 = v1 * (1 + (self.m - self.M) / (self.m + self.M))
            Mvx.append(v3)

            v1 = y_velocities[i]
            v2 = v1 * (self.m - self.M) / (self.m + self.M)
            v3 = v1 * (1 + (self.m - self.M) / (self.m + self.M))
            Mvy.append(v3)

        Vx = np.sum(Mvx)
        Vy = np.sum(Mvy)
        V = np.array([Vx, Vy])

        return V

# Function to simulate the velocity and drag force constant
def simulate_drag_force_constant(M):
    interactions = Interactions(M=M, m=1)
    velocity = np.array([5.0, 0.0])
    velocities = []
    drag_force_coefficient = []

    for t in range(100):
        particle_velocities = np.random.uniform(-2, 2, size=(n_particles, 2))
        
        final_velocity = interactions.collision(particle_velocities)

        drag_force_constant = np.linalg.norm(final_velocity)**2

        velocities.append(np.linalg.norm(final_velocity))
        drag_force_coefficient.append(drag_force_constant)

        velocity = final_velocity

    return velocities, drag_force_coefficient

# Plot multiple mass ratios on the same graph
def simulate_multiple_mass_ratios():
    mass_ratios = [0.5,1, 2, 10, 20, 50, 100]
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    plt.figure(figsize=(12, 8))

    for M, color in zip(mass_ratios, colors):
        velocities, drag_force_coefficient = simulate_drag_force_constant(M)

        sorted_indices = np.argsort(velocities)
        velocities_sorted = np.array(velocities)[sorted_indices]
        drag_force_coefficient_sorted = np.array(drag_force_coefficient)[sorted_indices]

        plt.scatter(velocities_sorted, drag_force_coefficient_sorted, color=color, alpha=0.6, marker='o', label=f'M = {M}')
        plt.plot(velocities_sorted, drag_force_coefficient_sorted, color=color, linestyle='-', linewidth=1)

    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Drag Force Constant (N/m/s)')
    plt.title('Velocity vs Drag Force Constant for Different Mass Ratios')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("velocity_vs_drag_force_constant_comparison.png")
    plt.show()

# Run the simulation for multiple mass ratios
simulate_multiple_mass_ratios()