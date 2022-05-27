import numpy as np
import matplotlib.pyplot as plt
import settings

class Error:
    def __init__(self, original_field, reconstructed_field, grid, intersections):
        self.original_field = original_field
        self.reconstructed_field = reconstructed_field
        self.grid = grid
        self.global_error = 0
        self.global_error_intersections = 0
        self.intersections = intersections

    def sample_fields(self, point):
        if settings.generate_your_own_measurement:
            original_vector = self.original_field.Sample(point[0], point[1], point[2])
        else:
            original_vector = None
        reconstructed_vector = self.reconstructed_field.SampleField(point)
        return(reconstructed_vector, original_vector)

    def Intersection_error(self):
        for i in range(len(self.intersections)):
            vector, vector_original = Error.sample_fields(self, self.intersections[i])
            u = vector[0]
            v = vector[1]
            w = vector[2]

            u_orig = vector_original[0]
            v_orig = vector_original[1]
            w_orig = vector_original[2]

            norm_v_original = (u_orig**2 + v_orig**2 + w_orig**2)**0.5
            if (norm_v_original != 0):
                error = ((u_orig-u)**2 + (v_orig - v)**2 + (w_orig - w)**2)**0.5 / (norm_v_original)
            else:
                error = 0
            self.global_error_intersections += error

        print('intersection error = ' + str(self.global_error_intersections * 100 / len(self.intersections)) + ' %')

    def Global_Error(self):
        res = settings.plot_interpolated_resolution

        x = np.linspace(self.grid.x_min +settings.interpolation_offset_x, self.grid.x_max - settings.interpolation_offset_x, res)
        y = np.linspace(self.grid.y_min + settings.interpolation_offset_y, self.grid.y_max - settings.interpolation_offset_y, res)
        z = np.linspace(self.grid.z_min + settings.interpolation_offset_z, self.grid.z_max - settings.interpolation_offset_z, res)


        u = np.zeros([res, res])
        v = np.zeros([res, res])
        w = np.zeros([res, res])

        u_orig = np.zeros([res, res])
        v_orig = np.zeros([res, res])
        w_orig = np.zeros([res, res])

        for i in range(res):
            for j in range(res):
                for k in range(res):
                    vector, vector_original = Error.sample_fields(self, np.array([x[i], y[j], z[k]]))
                    u[i][j] = vector[0]
                    v[i][j] = vector[1]
                    w[i][j] = vector[2]

                    u_orig[i][j] = vector_original[0]
                    v_orig[i][j] = vector_original[1]
                    w_orig[i][j] = vector_original[2]

                    norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                    if (norm_v_original != 0):
                        error = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                            (v[i][j] - v_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))/(norm_v_original)
                    else:
                        error = 0
                    self.global_error += error

        print('global error = ' + str(self.global_error*100/(res**3)) +' %')

    def SliceX(self, x):

        res = settings.plot_interpolated_resolution

        y = np.linspace(self.grid.y_min + settings.interpolation_offset_y, self.grid.y_max - settings.interpolation_offset_y, res)
        z = np.linspace(self.grid.z_min + settings.interpolation_offset_z, self.grid.z_max - settings.interpolation_offset_z, res)

        u = np.zeros([res, res])
        v = np.zeros([res, res])
        w = np.zeros([res, res])

        if settings.plot_error_sliced:
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

                if settings.plot_error_sliced:

                    u_orig[i][j] = vector_original[0]
                    v_orig[i][j] = vector_original[1]
                    w_orig[i][j] = vector_original[2]

                    norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                    if norm_v_original == 0:
                        norm_v_original += 0.01

                    if (norm_v_original != 0):
                        error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                (v[i][j] - v_orig[i][j]),
                                                                (w[i][j] - w_orig[i][j])])))/(norm_v_original)
        Error.ShowQuiver(self, y, z, v, w, x, 'x', [self.grid.y_min, self.grid.y_max], [self.grid.z_min, self.grid.z_max], 'y', 'z')

        if settings.plot_error_sliced:
            Error.ShowError(self, y, z, error, x, 'x', [self.grid.y_min, self.grid.y_max], [self.grid.z_min, self.grid.z_max], 'y', 'z')
            Error.ShowQuiver_original(self, y, z, v_orig, w_orig, x, 'x', [self.grid.y_min, self.grid.y_max], [self.grid.z_min, self.grid.z_max], 'y', 'z')

    def SliceY(self, y):

        res = settings.plot_interpolated_resolution

        x = np.linspace(self.grid.x_min +settings.interpolation_offset_x, self.grid.x_max - settings.interpolation_offset_x, res)
        z = np.linspace(self.grid.z_min +settings.interpolation_offset_z, self.grid.z_max - settings.interpolation_offset_z, res)

        u = np.zeros([res, res])
        v = np.zeros([res, res])
        w = np.zeros([res, res])

        if settings.plot_error_sliced:
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

                if settings.plot_error_sliced:

                    u_orig[i][j] = vector_original[0]
                    v_orig[i][j] = vector_original[1]
                    w_orig[i][j] = vector_original[2]

                    norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                    if norm_v_original == 0:
                        norm_v_original += 0.01

                    if (norm_v_original != 0):
                        error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                (v[i][j] - v_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))/(norm_v_original)
        Error.ShowQuiver(self, x, z, u, w, y, 'y', [self.grid.x_min, self.grid.x_max], [self.grid.z_min, self.grid.z_max], 'x', 'z')

        if settings.plot_error_sliced:
            Error.ShowError(self, x, z, error, y, 'y', [self.grid.x_min, self.grid.x_max], [self.grid.z_min, self.grid.z_max], 'x', 'z')
            Error.ShowQuiver_original(self, x, z, u_orig, w_orig, y, 'y', [self.grid.x_min, self.grid.x_max], [self.grid.z_min, self.grid.z_max], 'x', 'z')

    def SliceZ(self, z):

        res = settings.plot_interpolated_resolution

        x = np.linspace(self.grid.x_min + settings.interpolation_offset_x, self.grid.x_max - settings.interpolation_offset_x, res)
        y = np.linspace(self.grid.y_min + settings.interpolation_offset_y, self.grid.y_max - settings.interpolation_offset_y, res)

        u = np.zeros([res, res])
        v = np.zeros([res, res])
        w = np.zeros([res, res])

        if settings.plot_error_sliced:
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

                if settings.plot_error_sliced:
                    u_orig[i][j] = vector_original[0]
                    v_orig[i][j] = vector_original[1]
                    w_orig[i][j] = vector_original[2]

                    norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                    if norm_v_original == 0:
                        norm_v_original += 0.01

                    if (norm_v_original != 0):
                        error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                (v[i][j] - v_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))/(norm_v_original)
        Error.ShowQuiver(self, x, y, u, v, z, 'z', [self.grid.x_min, self.grid.x_max], [self.grid.y_min, self.grid.y_max], 'x', 'y')

        if settings.plot_error_sliced:
            Error.ShowError(self, x, y, error, z, 'z',  [self.grid.x_min, self.grid.x_max], [self.grid.y_min, self.grid.y_max], 'x', 'y')
            Error.ShowQuiver_original(self, x, y, u_orig, v_orig, z, 'z', [self.grid.x_min, self.grid.x_max], [self.grid.y_min, self.grid.y_max], 'x', 'y')


    def ShowError(self, x, y, error, height, axis, axis1_lim, axis2_lim, axis1_name, axis2_name):
        fig1 = plt.figure(figsize=(15, 15))
        img1 = plt.contourf(x, y, error, 100)
        plt.xlim(axis1_lim)
        plt.ylim(axis2_lim)
        plt.xlabel(axis1_name + '-axis')
        plt.ylabel(axis2_name + '-axis')
        fig1.colorbar(img1)
        plt.title('Relative error at ' + axis + ' =' + str(height))
        if settings.save_figures:
            plt.savefig('..\Output\calculations_'+settings.Name_of_calculation +'\Error_at ' + axis + '= ' + str(height)+'.jpeg', format='jpeg')
        plt.show()

    def ShowQuiver(self, x, y, u, v, height, axis, axis1_lim, axis2_lim, axis1_name, axis2_name):
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(15, 15))
        plt.xlim(axis1_lim)
        plt.ylim(axis2_lim)
        plt.style.use('fivethirtyeight')
        plt.xlabel(axis1_name + '-axis')
        plt.ylabel(axis2_name + '-axis')
        plt.quiver(X, Y, v, u, scale= settings.quiver_scale)
        plt.title('Reconstructed Field at ' + axis + '= ' + str(height))
        plt.gca().set_aspect('equal', adjustable='box')
        if settings.save_figures:
            plt.savefig('..\Output\calculations_'+settings.Name_of_calculation +'\Reconstructed_Field_at ' + axis + '= ' + str(height)+'.jpeg', format='jpeg')
        plt.show()

    def ShowQuiver_original(self, x, y, u, v, height, axis, axis1_lim, axis2_lim, axis1_name, axis2_name):
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(15, 15))
        plt.xlim(axis1_lim)
        plt.ylim(axis2_lim)
        plt.style.use('fivethirtyeight')
        plt.xlabel(axis1_name + '-axis')
        plt.ylabel(axis2_name + '-axis')
        plt.quiver(X, Y, v, u)
        plt.title('Original Field at ' + axis + '= ' + str(height))
        plt.gca().set_aspect('equal', adjustable='box')
        if settings.save_figures:
            plt.savefig('..\Output\calculations_'+settings.Name_of_calculation +'\Original_Field_at ' + axis + '= ' + str(height) +'.jpeg', format='jpeg')
        plt.show()
