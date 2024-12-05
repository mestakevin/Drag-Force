import numpy as np
import matplotlib.pyplot as plt

# Constants for the simulation
m = 1  # Mass of each particle (kg)
dt = 0.1  # Time step for the simulation (s)
#n_particles = 100  # Number of particles
speed_range = 5.0  # Speed range for initial particle velocities
box_size = 10.0  # Size of the simulation box

# Prompt the user for the value of M (how many m the sphere's mass equals)
M = int(input("Enter the number of particles (M) equivalent to the mass of the sphere (M = how many m): "))
n_particles = int(input("Enter the number of particles): "))

# Particle class (remains the same as your code)
class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.history = [position.copy()]

# Interactions class for collision and velocity change estimation
class Interactions:
    def __init__(self, M, m):
        self.M = M  # Mass of the sphere in terms of particles' masses
        self.m = m  # Mass of each particle

    def collision(self, velocities):  # velocities should be in [x, y] components for each particle
        x_velocities = velocities[:, 0]  # x_velocities of particles
        y_velocities = velocities[:, 1]  # y_velocities of particles
        velocities = []
        Mvx = []  # Final x-velocity of the sphere
        Mvy = []  # Final y-velocity of the sphere
        for i in range(len(x_velocities)):
            # Using conservation of momentum (elastic collisions)
            v1 = x_velocities[i]  # Initial x_velocity of particle
            v2 = v1 * (self.m - self.M) / (self.m + self.M)  # Final x_velocity of particle
            v3 = v1 * (1 + (self.m - self.M) / (self.m + self.M))  # Final x_velocity of sphere
            x_v = [v2]
            Mvx.append(v3)

            v1 = y_velocities[i]  # Initial y_velocity of particle
            v2 = v1 * (self.m - self.M) / (self.m + self.M)  # Final y_velocity of particle
            v3 = v1 * (1 + (self.m - self.M) / (self.m + self.M))  # Final y_velocity of sphere
            y_v = [v2]
            Mvy.append(v3)

            velocities.append([x_v, y_v])

        # Calculate the total final velocity of the sphere by summing the velocities of particles
        Vx = np.sum(Mvx)
        Vy = np.sum(Mvy)
        V = np.array([Vx, Vy])

        return velocities, V

# Function to simulate the velocity and drag force constant over time for the sphere, considering temperature and air density
def simulate_drag_force_constant_with_density():
    # Constants for air and environment
    P = 101325  # Pressure in Pascals (constant, sea level)
    R = 287.05  # Specific gas constant for air (J/kg·K)
    T = 300  # Temperature in Kelvin (initial value)

    # Initialize interactions and velocity
    interactions = Interactions(M=M, m=1)
    velocity = np.array([5.0, 0.0])  # Initial velocity of the sphere (m/s)
    velocities = []  # List to store velocities for plotting
    drag_force_coefficient = []  # List to store drag force constants for plotting
    air_density_values = []  # To store air density values

    # Simulate the motion of the sphere under drag force
    for t in range(100):
        # Calculate air density based on temperature
        air_density = P / (R * T)

        # Simulate a random set of velocities for the particles
        particle_velocities = np.random.uniform(-2, 2, size=(n_particles, 2))  # Random particle velocities
        
        # Calculate the new velocities of particles and sphere using collision
        _, final_velocity = interactions.collision(particle_velocities)

        # Calculate the velocity change of the sphere
        delta_velocity = np.linalg.norm(final_velocity - velocity)

        # Calculate drag force constant
        # Drag force coefficient = 0.5 * C_d * rho * A (simplified as proportional to rho * v^2)
        drag_force_constant = air_density * np.linalg.norm(final_velocity)**2

        # Store the velocity, air density, and drag force constant for plotting
        velocities.append(np.linalg.norm(final_velocity))
        drag_force_coefficient.append(drag_force_constant)
        air_density_values.append(air_density)

        # Update velocity and temperature (simulate cooling or heating)
        velocity = final_velocity
        T -= 0.5  # Example: Decrease temperature over time (cooling)

    # Sort the data points by velocity
    sorted_indices = np.argsort(velocities)
    velocities_sorted = np.array(velocities)[sorted_indices]
    drag_force_coefficient_sorted = np.array(drag_force_coefficient)[sorted_indices]
    air_density_sorted = np.array(air_density_values)[sorted_indices]

    # Plot the velocity vs drag force constant with air density and temperature
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(velocities_sorted, drag_force_coefficient_sorted, c=air_density_sorted, cmap='viridis', marker='o', alpha=0.7)
    plt.colorbar(scatter, label='Temperature coefficient (K)')
    plt.plot(velocities_sorted, drag_force_coefficient_sorted, color='darkorange')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Drag Force Constant (N·s/m²)')
    plt.title(f'Velocity vs Drag Force Constant (Mass = {M}, T varies)')
    plt.grid(True)

    # Save the plot with the dynamic filename
    filename = f"velocity_vs_drag_force_constant_with_density_M_{M}.png"
    plt.savefig(filename)
    plt.show()

# Run the simulation
simulate_drag_force_constant_with_density()
