from scipy.spatial import distance
import numpy as np


class wumDist:
    def __init__(self, u1, u2):
        self.u1 = np.array(u1)
        self.u2 = np.array(u2)

    def get_u1(self):
        return self.u1

    def get_u2(self):
        return self.u2

    def prototypes(self):
        u1_prototype = np.sum(self.u1) / self.u1.size
        u2_prototype = np.sum(self.u2) / self.u2.size

        return u1_prototype, u2_prototype

    def prt(self):
        p1, p2 = self.prototypes()
        return 1 / distance.cosine(p1, p2)

    def apd(self):
        return np.NaN

    def jsd(self):
        return np.NaN

    def div(self):
        p1, p2 = self.prototypes()
        dists_from_prototype_1 = np.array([distance.cosine(vec, p1) for vec in self.u1])
        dists_from_prototype_2 = np.array([distance.cosine(vec, p2) for vec in self.u2])
        var_coefficient_1 = np.sum(dists_from_prototype_1) / self.u1_size
        var_coefficient_2 = np.sum(dists_from_prototype_2) / self.u2.size

        return abs(var_coefficient_1 - var_coefficient_2)
