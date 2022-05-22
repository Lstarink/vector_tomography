import numpy as np
import matplotlib.pyplot as plt
import settings

class Error:
    def __init__(self, original_field, reconstructed_field, grid):
        self.original_field = original_field
        self.reconstructed_field = reconstructed_field
        self.grid = grid

    def sample_fields(self, point):
        original_vector = self.original_field.Sample(point[0], point[1], point[2])
        reconstructed_vector = self.reconstructed_field.SampleField(point)
        return(original_vector, reconstructed_vector)


    def SliceX(self, x):

        res = settings.plot_interpolated_resolution

        y = np.linspace(self.grid.y_min, self.grid.y_max, res)
        z = np.linspace(self.grid.z_min, self.grid.z_max, res)

        u = np.zeros([res, res])
        v = np.zeros([res, res])
        w = np.zeros([res, res])

        u_orig = np.zeros([res, res])
        v_orig = np.zeros([res, res])
        w_orig = np.zeros([res, res])

        error = np.zeros([res, res])

        for i in range(res):
            for j in range(res):
                vector, vector_original = Error.sample_fields(self, np.array([x, y[i], z[j]]))
                u[i][j] = vector[0]
                v[i][j] = vector[1]
                w[i][j] = vector[2]

                u_orig[i][j] = vector_original[0]
                v_orig[i][j] = vector_original[1]
                w_orig[i][j] = vector_original[2]

                norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                # TODO find something prettier for this
                if norm_v_original == 0:
                    norm_v_original += 0.01

                if (norm_v_original != 0):
                    error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                            (v[i][j] - v_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))# / (norm_v_original)

        Error.ShowError(self, y, z, error, x, 'x')
        Error.ShowQuiver(self, y, z, v, w, x, 'x')
        Error.ShowQuiver_original(self, y, z, v_orig, w_orig, x, 'x')

    def SliceY(self, y):

        res = settings.plot_interpolated_resolution

        x = np.linspace(self.grid.x_min, self.grid.x_max, res)
        z = np.linspace(self.grid.z_min, self.grid.z_max, res)

        u = np.zeros([res, res])
        v = np.zeros([res, res])
        w = np.zeros([res, res])

        u_orig = np.zeros([res, res])
        v_orig = np.zeros([res, res])
        w_orig = np.zeros([res, res])

        error = np.zeros([res, res])

        for i in range(res):
            for j in range(res):
                vector, vector_original = Error.sample_fields(self, np.array([x[i], y, z[j]]))
                u[i][j] = vector[0]
                v[i][j] = vector[1]
                w[i][j] = vector[2]

                u_orig[i][j] = vector_original[0]
                v_orig[i][j] = vector_original[1]
                w_orig[i][j] = vector_original[2]

                norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                # TODO find something prettier for this
                if norm_v_original == 0:
                    norm_v_original += 0.01

                if (norm_v_original != 0):
                    error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                            (v[i][j] - v_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))# / (norm_v_original)

        Error.ShowError(self, x, z, error, y, 'y')
        Error.ShowQuiver(self, x, z, u, w, y, 'y')
        Error.ShowQuiver_original(self, x, z, u_orig, w_orig, y, 'y')

    def SliceZ(self, z):

        res = settings.plot_interpolated_resolution

        x = np.linspace(self.grid.x_min, self.grid.x_max, res)
        y = np.linspace(self.grid.y_min, self.grid.y_max, res)

        u = np.zeros([res, res])
        v = np.zeros([res, res])
        w = np.zeros([res, res])

        u_orig = np.zeros([res, res])
        v_orig = np.zeros([res, res])
        w_orig = np.zeros([res, res])

        error = np.zeros([res, res])

        for i in range(res):
            for j in range(res):
                vector, vector_original = Error.sample_fields(self, np.array([x[i], y[j], z]))
                u[i][j] = vector[0]
                v[i][j] = vector[1]
                w[i][j] = vector[2]

                u_orig[i][j] = vector_original[0]
                v_orig[i][j] = vector_original[1]
                w_orig[i][j] = vector_original[2]

                norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                # TODO find something prettier for this
                if norm_v_original == 0:
                    norm_v_original += 0.01

                if (norm_v_original != 0):
                    error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                            (v[i][j] - v_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))# / (norm_v_original)

        Error.ShowError(self, x, y, error, z, 'z')
        Error.ShowQuiver(self, x, y, u, v, z, 'z')
        Error.ShowQuiver_original(self, x, y, u_orig, v_orig, z, 'z')


    def ShowError(self, x, y, error, height, axis):
        fig1 = plt.figure(figsize=(15, 15))
        img1 = plt.contourf(x, y, error, 100)
        fig1.colorbar(img1)
        plt.title('Relative in plane error at ' + axis + ' =' + str(height))
        plt.show()

    def ShowQuiver(self, x, y, u, v, height, axis):
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(15, 15))
        plt.quiver(X, Y, v, u)
        plt.title('Reconstructed Field at ' + axis + '= ' + str(height))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def ShowQuiver_original(self, x, y, u, v, height, axis):
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(15, 15))
        plt.quiver(X, Y, v, u)
        plt.title('Original Field at ' + axis + '= ' + str(height))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
