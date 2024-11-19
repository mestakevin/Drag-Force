# Particle Class and so on
class particle:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.mass = 1
        self.radius = 0
        self.eng = self.getEng()

    def getEng(self):
        return 0.5 * self.mass * self.getSpeed() ** 2

    def updEng(self, new_eng):
        self.eng = new_eng

    def getVel(self):
        return np.linalg.norm(self.getVel())
    
    def updVel(self, new_vel):
        self.vel = new_vel

    def getPos(self):
        return self.pos

    def getRadius(self):
        return self.radius
