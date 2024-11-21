import numpy as np
import matplotlib.pyplot as plt

# Constants for the simulation
m = 1  # Mass of each particle (kg)
dt = 0.1  # Time step for the simulation (s)
n_particles = 100  # Number of particles
speed_range = 5.0  # Speed range for initial particle velocities
box_size = 10.0  # Size of the simulation box

# Prompt the user for the value of M (how many m the sphere's mass equals)
M = int(input("Enter the number of particles (M) equivalent to the mass of the sphere (M = how many m): "))

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

# ParticleSimulation class for the movement of particles
class ParticleSimulation:
    def __init__(self, n_particles: int, box_size: float = 10.0, speed_range: float = 2.0):
        self.box_size = box_size
        self.particles = []
        for _ in range(n_particles):
            position = np.random.uniform(0, box_size, 2)
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0, speed_range)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
            self.particles.append(Particle(position, velocity))
    
    def update(self, dt: float = 0.1):
        """Update positions of particles and apply wall collisions"""
        for particle in self.particles:
            particle.position += particle.velocity * dt
            for i in range(2):
                if particle.position[i] <= 0 or particle.position[i] >= self.box_size:
                    particle.position[i] = np.clip(particle.position[i], 0, self.box_size)
                    particle.velocity[i] *= -1
            particle.history.append(particle.position.copy())
    
    def simulate(self, steps: int = 100):
        """Run the simulation for a given number of steps"""
        for _ in range(steps):
            self.update()

# Function to simulate the velocity and drag force constant over time for the sphere
def simulate_drag_force_constant():
    # Initialize interactions and velocity
    interactions = Interactions(M=M, m=1)
    velocity = np.array([5.0, 0.0])  # Initial velocity of the sphere (m/s)
    velocities = []  # List to store velocities for plotting
    drag_force_coefficient = []  # List to store drag force constants for plotting
    time = 0  # Initial time

    # Simulate the motion of the sphere under drag force
    for t in range(100):
        # Simulate a random set of velocities for the particles
        particle_velocities = np.random.uniform(-2, 2, size=(n_particles, 2))  # Random particle velocities
        
        # Calculate the new velocities of particles and sphere using collision
        _, final_velocity = interactions.collision(particle_velocities)

        # Calculate the velocity change of the sphere
        delta_velocity = np.linalg.norm(final_velocity - velocity)

        # Calculate drag force constant: assume it scales with the velocity squared (common model for drag)
        # Drag force coefficient = 0.5 * C_d * rho * A (approximation)
        # Here we simulate it as proportional to velocity squared
        drag_force_constant = np.linalg.norm(final_velocity)**2  # This is an assumption for drag constant

        # Store the velocity and drag force constant for plotting
        velocities.append(np.linalg.norm(final_velocity))
        drag_force_coefficient.append(drag_force_constant)

        # Update velocity for the next iteration
        velocity = final_velocity

    # Sort the data points by velocity to avoid back-and-forth lines
    sorted_indices = np.argsort(velocities)
    velocities_sorted = np.array(velocities)[sorted_indices]
    drag_force_coefficient_sorted = np.array(drag_force_coefficient)[sorted_indices]

    # Plot the velocity vs drag force constant (smooth curve)
    plt.figure(figsize=(10, 6))
    plt.scatter(velocities_sorted, drag_force_coefficient_sorted, color='blue', alpha=0.6, marker='o')
    plt.plot(velocities_sorted, drag_force_coefficient_sorted, color='darkorange', linestyle='-', linewidth=1)  # Connect the dots with a line
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Drag Force Constant (N/m/s)')
    plt.title(f'Velocity vs Drag Force Constant (Mass = {M})mass')
    plt.grid(True)

    # Save the plot with the dynamic filename
    filename = f"velocity_vs_drag_force_constant_M_{M}.png"
    plt.savefig(filename)
    plt.show()

# Run the simulation
simulate_drag_force_constant()
