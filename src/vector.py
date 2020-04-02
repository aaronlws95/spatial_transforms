import numpy as np

class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def numpy(self):
        return np.asarray([self.x, self.y, self.z])