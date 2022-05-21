import numpy as np

class Intersection:
    def __init__(self, location):
        self.location = location
        self.rank = 0
        self.tubes_present = 0
        self.vector = []

    def add_tube(self):
        self.tubes_present += 1

    def add_vector(self, vector):
        self.vector.append(vector)

    def determine_rank(self):
        vector_space = np.zeros([3, self.tubes_present])
        for i in range(len(self.vector)):
            for j in range(3):
                vector_space[j][i] = self.vector[i][j]
        self.rank = np.linalg.matrix_rank(vector_space)



