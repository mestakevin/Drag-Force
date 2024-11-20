import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple

class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.history = [position.copy()]
        
class ParticleSimulation:
    def __init__(self, n_particles: int, box_size: float = 10.0, speed_range: float = 2.0):
        self.box_size = box_size
        self.particles: List[Particle] = []
        
        # Initialize particles with random positions and velocities
        for _ in range(n_particles):
            position = np.random.uniform(0, box_size, 2)
            # Random angle and speed for velocity
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0, speed_range)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
            
            self.particles.append(Particle(position, velocity))
    
    def update(self, dt: float = 0.1):
        """Update positions and wall collisions"""
        for particle in self.particles:
            # Update position
            particle.position += particle.velocity * dt
            
            # wall collisions
            for i in range(2):  # Check both x and y coordinates
                if particle.position[i] <= 0:
                    particle.position[i] = 0
                    particle.velocity[i] *= -1
                elif particle.position[i] >= self.box_size:
                    particle.position[i] = self.box_size
                    particle.velocity[i] *= -1
            
            # Store position history
            particle.history.append(particle.position.copy())
    
    def simulate(self, steps: int = 100):
        """Run simulation for given number of steps"""
        for _ in range(steps):
            self.update()
    
    def plot_trajectories(self):
        """Plot particle trajectories"""
        plt.figure(figsize=(10, 10))
        
        # Plot box boundaries
        plt.plot([0, self.box_size, self.box_size, 0, 0], 
                [0, 0, self.box_size, self.box_size, 0], 
                'k-', linewidth=2)
        
        # Plot trajectories
        for particle in self.particles:
            history = np.array(particle.history)
            plt.plot(history[:, 0], history[:, 1], '-', linewidth=0.5, alpha=0.5)
            # Plot final position
            plt.plot(history[-1, 0], history[-1, 1], 'o')
        
        plt.xlim(-1, self.box_size + 1)
        plt.ylim(-1, self.box_size + 1)
        plt.title(f'Particle Trajectories (n={len(self.particles)})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def animate(self, steps: int = 100):
        """Create animation of particle motion"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Initialize scatter plot
        scatter = ax.scatter([], [], c='b', alpha=0.6)
        
        # Set plot limits
        ax.set_xlim(-1, self.box_size + 1)
        ax.set_ylim(-1, self.box_size + 1)
        ax.grid(True)
        
        # Plot box boundaries
        ax.plot([0, self.box_size, self.box_size, 0, 0], 
                [0, 0, self.box_size, self.box_size, 0], 
                'k-', linewidth=2)
        
        def init():
            scatter.set_offsets(np.c_[[], []])
            return scatter,
        
        def animate(frame):
            self.update()
            positions = np.array([p.position for p in self.particles])
            scatter.set_offsets(positions)
            return scatter,
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=steps,
                           interval=50, blit=True)
        plt.show()
        return anim

simulation = ParticleSimulation(n_particles=100)
simulation.simulate(steps=200)
simulation.plot_trajectories()

simulation = ParticleSimulation(n_particles=100)
anim = simulation.animate(steps=200)
