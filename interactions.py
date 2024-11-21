import numpy as np

class interactions:
    global M, m, V
    
    M = 100  # Mass of sphere
    m = 1  # Mass of each particle

    def collision(velocities):  # velocities should be in [x,y] components for each particle
        '''
        To calculate the momentum exchange for each particle. The sphere is assumed stationary at the instance of collision.
        '''
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
            
            # in x-direction
            v1 = x_velocities[i]  # initial x_velocity of particle
            v2 = v1*(m-M)/(m+M)  # final x_velocity of particle
            v3 = v1*(1+(m-M)/(m+M))  # final x_velocity of sphere
            x_v.append(v2)
            Mvx.append(v3)
 
            # in y-direction
            v1 = y_velocities[i]  # initial y_velocity of particle
            v2 = v1*(m-M)/(m+M)  # final y_velocity of particle
            v3 = v1*(1+(m-M)/(m+M))  # final y_velocity of sphere
            y_v.append(v2)
            Mvy.append(v3)
            
            velocities.append([x_v,y_v])
            i += 1

        
        Vx = np.sum(Mvx)
        Vy = np.sum(Mvy)
        V = [Vx,Vy]
        
        return velocities, V

    def drag(Vi,dt):  # Vi is the initial 2D velocity vector of the sphere, dt is the time step (s).
        '''
        To calculate the force introduced by interactions.
        '''
        vi = Vi
        vf = V
        F = M*(vf-vi)/dt  
        Fd = np.dot(F,vi/abs(vi))
        return Fd
