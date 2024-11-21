import numpy as np

class interactions:
    global M, m
    M = 100  # Mass of sphere
    m = 1  # Mass of each particle

    def collision(velocities):  # velocities should be in [x,y] components
        x_velocities = velocities[:,0]  # x_velocities of particles
        y_velocities = velocities[:,1]  # y_velocities of particles
        velocities = []
        i = 0
        Mvx = []
        Mvy = []
        while i < len(x_velocities):
            # using conservation of momentum and kinetic energy assuming a purely elastic collision
            x_v = []
            y_v = []
            
            # x-direction
            v1 = x_velocities[i]
            v2 = v1*(m-M)/(m+M)
            v3 = v1*(1+(m-M)/(m+M))
            x_v.append(v2)
            Mvx.append(v3)
            
            # y-direction
            v1 = y_velocities[i]
            v2 = v1*(m-M)/(m+M)
            v3 = v1*(1+(m-M)/(m+M))
            y_v.append(v2)
            Mvy.append(v3)
            
            velocities.append([x_v,y_v])
            i += 1
            
        Vx = np.sum(Mvx)
        Vy = np.sum(Mvy)
        V = [Vx,Vy]
        return velocities, V
