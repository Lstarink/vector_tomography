import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import settings

class Error:
    def __init__(self, original_field, reconstructed_field, grid, intersections):
        self.original_field = original_field
        self.reconstructed_field = reconstructed_field
        self.grid = grid
        self.global_error = 0
        self.total_yz_error = 0
        self.total_xz_error = 0
        self.total_xy_error = 0
        self.global_error_intersections = 0
        self.intersections = intersections

    def sample_fields(self, point):
        if settings.generate_your_own_measurement:
            original_vector = self.original_field.Sample(point[0], point[1], point[2])
            #print('vector at:', point[0],'    ', point[1], '  ', point[2],' is ',  original_vector)
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

        #print('intersection error = ' + str(self.global_error_intersections * 100 / len(self.intersections)) + ' %')
        print(str(self.global_error_intersections * 100 / len(self.intersections)) + ' %')

    def Global_Error(self):
        res = settings.plot_interpolated_resolution

        x = np.linspace(self.grid.x_min +settings.interpolation_offset_x, self.grid.x_max - settings.interpolation_offset_x, res)
        y = np.linspace(self.grid.y_min + settings.interpolation_offset_y, self.grid.y_max - settings.interpolation_offset_y, res)
        z = np.linspace(self.grid.z_min + settings.interpolation_offset_z, self.grid.z_max - settings.interpolation_offset_z, res)

        error_list = []

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
                    error_list.append(100*error)
                    self.global_error += error

                    yz_norm_original = np.linalg.norm(np.array([v_orig[i][j]], w_orig[i][j]))
                    if yz_norm_original != 0:
                        yz_error = (np.linalg.norm(np.array([(v[i][j] - v_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))/(yz_norm_original)
                    else:
                        yz_error = 0

                    self.total_yz_error += yz_error

                    xz_norm_original = np.linalg.norm(np.array([u_orig[i][j]], w_orig[i][j]))
                    if xz_norm_original != 0:
                        xz_error = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                            (w[i][j] - w_orig[i][j])])))/(xz_norm_original)
                    else:
                        xz_error = 0

                    self.total_xz_error += xz_error

                    xy_norm_original = np.linalg.norm(np.array([u_orig[i][j]], v_orig[i][j]))
                    if xy_norm_original != 0:
                        xy_error = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                            (v[i][j] - v_orig[i][j])])))/(xy_norm_original)
                    else:
                        xy_error = 0

                    self.total_xy_error += xy_error

        #print('global error = ' + str(self.global_error*100/(res**3)) +' %')
        print(str(self.global_error*100/(res**3)) +' %')
        plt.figure()
        plt.hist(error_list,bins=100)
        plt.savefig('..\Output\calculations_' + settings.Name_of_calculation + '\Plots\Filtered_error100' + '.jpeg',
                    format='jpeg')

        print('xz_error', self.total_xz_error*100/(res**3), '%')
        print('yz_error', self.total_yz_error*100/(res**3), '%')
        print('xz_error', self.total_xz_error*100/(res**3), '%')


        error_list.sort()
        filtered_error = error_list[0: int(0.9*res**3)]

        filtered_global_error = sum(filtered_error)/(0.9*res**3)
        #print('filtered_global_error90 = ' + str(filtered_global_error) + ' %')
        print(str(filtered_global_error) + ' %')
        plt.figure()
        plt.hist(filtered_error,bins=100)
        plt.savefig('..\Output\calculations_' + settings.Name_of_calculation + '\Plots\Filtered_error90' + '.jpeg',
                    format='jpeg')

        filtered_error2 = error_list[0: int(0.75*res**3)]

        filtered_global_error2 = sum(filtered_error2)/(0.9*res**3)
        #print('filtered_global_error = ' + str(filtered_global_error2) + ' %')
        print(str(filtered_global_error2) + ' %')
        plt.figure()
        plt.hist(filtered_error2,bins=100)
        plt.savefig('..\Output\calculations_' + settings.Name_of_calculation + '\Plots\Filtered_error75' + '.jpeg',
                    format='jpeg')


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

                    if settings.inplane_error:
                        norm_v_original = np.linalg.norm(np.array([v_orig[i][j], w_orig[i][j]]))
                        if norm_v_original == 0:
                            norm_v_original += 0.01
                        if (norm_v_original != 0):
                            error[i][j] = (np.linalg.norm(np.array([(v[i][j] - v_orig[i][j]),
                                                                    (w[i][j] - w_orig[i][j])]))) / (norm_v_original)

                    else:
                        norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                        if norm_v_original == 0:
                            norm_v_original += 0.01

                        if (norm_v_original != 0):
                            error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                    (v[i][j] - v_orig[i][j]),
                                                                    (w[i][j] - w_orig[i][j])]))) / (norm_v_original)

        Error.ShowQuiver(self, y, z, v, w, x, 'x', [self.grid.y_min, self.grid.y_max], [self.grid.z_min, self.grid.z_max], 'y', 'z')

        if settings.plot_error_sliced:
            if settings.only_combined:
                Error.ShowAll(self, y, z, v, w, v_orig, w_orig, error, x, 'x', [self.grid.y_min, self.grid.y_max],
                              [self.grid.z_min, self.grid.z_max], 'y', 'z')
            else:
                Error.ShowError(self, y, z, error, x, 'x', [self.grid.y_min, self.grid.y_max], [self.grid.z_min, self.grid.z_max], 'y', 'z')
                Error.ShowQuiver_original(self, y, z, v_orig, w_orig, x, 'x', [self.grid.y_min, self.grid.y_max], [self.grid.z_min, self.grid.z_max], 'y', 'z')
                Error.ShowAll(self, y, z, v, w, v_orig, w_orig, error, x, 'x', [self.grid.y_min, self.grid.y_max],
                              [self.grid.z_min, self.grid.z_max], 'y', 'z')
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

                    if settings.inplane_error:
                        norm_v_original = np.linalg.norm(np.array([u_orig[i][j], w_orig[i][j]]))
                        if norm_v_original == 0:
                            norm_v_original += 0.01
                        if (norm_v_original != 0):
                            error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                    (w[i][j] - w_orig[i][j])])))/(norm_v_original)

                    else:
                        norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                        if norm_v_original == 0:
                            norm_v_original += 0.01

                        if (norm_v_original != 0):
                            error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                    (v[i][j] - v_orig[i][j]),
                                                                (w[i][j] - w_orig[i][j])])))/(norm_v_original)

        Error.ShowQuiver(self, x, z, u, w, y, 'y', [self.grid.x_min, self.grid.x_max], [self.grid.z_min, self.grid.z_max], 'x', 'z')
        if settings.plot_error_sliced:
            if settings.only_combined:
                Error.ShowAll(self, x, z, u, w, u_orig, w_orig, error, y, 'y', [self.grid.x_min, self.grid.x_max],
                              [self.grid.z_min, self.grid.z_max], 'x', 'z')
            else:

                Error.ShowError(self, x, z, error, y, 'y', [self.grid.x_min, self.grid.x_max], [self.grid.z_min, self.grid.z_max], 'x', 'z')
                Error.ShowQuiver_original(self, x, z, u_orig, w_orig, y, 'y', [self.grid.x_min, self.grid.x_max], [self.grid.z_min, self.grid.z_max], 'x', 'z')
                Error.ShowAll(self, x, z, u, w, u_orig, w_orig, error, y, 'y', [self.grid.x_min, self.grid.x_max],
                              [self.grid.z_min, self.grid.z_max], 'x', 'z')

    def SliceZ(self, z):

        global error
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

                    if settings.inplane_error:
                        norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j]]))
                        if norm_v_original == 0:
                            norm_v_original += 0.01
                        if (norm_v_original != 0):
                            error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                    (v[i][j] - v_orig[i][j])])))/(norm_v_original)

                    else:
                        norm_v_original = np.linalg.norm(np.array([u_orig[i][j], v_orig[i][j], w_orig[i][j]]))
                        if norm_v_original == 0:
                            norm_v_original += 0.01

                        if (norm_v_original != 0):
                            error[i][j] = (np.linalg.norm(np.array([(u[i][j] - u_orig[i][j]),
                                                                    (v[i][j] - v_orig[i][j]),
                                                                (w[i][j] - w_orig[i][j])])))/(norm_v_original)



        Error.ShowQuiver(self, x, y, u, v, z, 'z', [self.grid.x_min, self.grid.x_max], [self.grid.y_min, self.grid.y_max], 'x', 'y')
        if settings.plot_error_sliced:
            if settings.only_combined:
                Error.ShowAll(self, x, y, u, v, u_orig, v_orig, error, z, 'z', [self.grid.x_min, self.grid.x_max],
                              [self.grid.y_min, self.grid.y_max], 'x', 'y')
            else:

                Error.ShowError(self, x, y, error, z, 'z',  [self.grid.x_min, self.grid.x_max], [self.grid.y_min, self.grid.y_max], 'x', 'y')
                Error.ShowQuiver_original(self, x, y, u_orig, v_orig, z, 'z', [self.grid.x_min, self.grid.x_max], [self.grid.y_min, self.grid.y_max], 'x', 'y')
                Error.ShowAll(self, x, y, u, v, u_orig, v_orig, error, z, 'z', [self.grid.x_min, self.grid.x_max], [self.grid.y_min, self.grid.y_max], 'x', 'y')

    def ShowError(self, x, y, error, height, axis, axis1_lim, axis2_lim, axis1_name, axis2_name):
        error_ = error.transpose()
        fig1 = plt.figure(figsize=(15, 15))
        img1 = plt.contourf(x, y, error_, 100)
        plt.xlim(axis1_lim)
        plt.ylim(axis2_lim)
        plt.xlabel(axis1_name + '-axis')
        plt.ylabel(axis2_name + '-axis')
        fig1.colorbar(img1)
        plt.title('Relative error at ' + axis + ' =' + str(height))
        if settings.save_figures:
            plt.savefig('..\Output\calculations_'+settings.Name_of_calculation +'\Plots\Error_at ' + axis + '= ' + (str(height).replace('.', ',')) +'.jpeg', format='jpeg')
        if settings.show_sliced:
            plt.show()

    def ShowQuiver(self, x, y, u, v, height, axis, axis1_lim, axis2_lim, axis1_name, axis2_name):
        X, Y = np.meshgrid(x, y)
        u_ = np.transpose(u)
        v_ = np.transpose(v)
        plt.figure(figsize=(15, 15))
        plt.xlim(axis1_lim)
        plt.ylim(axis2_lim)
        plt.xlabel(axis1_name + '-axis')
        plt.ylabel(axis2_name + '-axis')
        plt.quiver(X, Y, u_, v_, scale= settings.quiver_scale, angles='xy')
        plt.streamplot(X, Y, u_, v_, linewidth=1)
        plt.title('Reconstructed Field at ' + axis + '= ' + str(round(height,3)))
        plt.gca().set_aspect('equal', adjustable='box')
        if settings.save_figures:
            plt.savefig('..\Output\calculations_'+settings.Name_of_calculation +'\Plots\Reconstructed_Field_at ' + axis + '= ' + (str(height).replace('.', ','))+'.jpeg', format='jpeg')
        if settings.show_sliced:
            plt.show()

    def ShowQuiver_original(self, x, y, u, v, height, axis, axis1_lim, axis2_lim, axis1_name, axis2_name):
        X, Y = np.meshgrid(x, y)
        u_ = np.transpose(u)
        v_ = np.transpose(v)
        plt.figure(figsize=(15, 15))
        plt.xlim(axis1_lim)
        plt.ylim(axis2_lim)
        plt.xlabel(axis1_name + '-axis')
        plt.ylabel(axis2_name + '-axis')
        plt.quiver(x, y, u_, v_, angles='xy')
        plt.streamplot(X, Y, u_, v_,linewidth=1)
        plt.title('Original Field at ' + axis + '= ' + str(height))
        plt.gca().set_aspect('equal', adjustable='box')
        if settings.save_figures:
            plt.savefig('..\Output\calculations_'+settings.Name_of_calculation +'\Plots\Original_Field_at ' + axis + '= ' + (str(height).replace('.', ',')) +'.jpeg', format='jpeg')
        if settings.show_sliced:
            plt.show()

    def ShowAll(self, x, y, u, v, u_orig, v_orig, error, height, axis, axis1_lim, axis2_lim, axis1_name, axis2_name):
        u_ = np.transpose(u)
        v_ = np.transpose(v)
        u_orig_ = np.transpose(u_orig)
        v_orig_ = np.transpose(v_orig)
        error_ = error.transpose()

        fig, (ax2, ax1, ax3) = plt.subplots(1,3,figsize=(45,15))
        fig.suptitle('Original and reconstructed field at ' + axis + '= ' + str(height), fontsize=35)

        ax1.title.set_text('Reconstructed Field')
        ax1.quiver(x, y, u_, v_, angles='xy')
        ax1.streamplot(x,y,u_,v_,linewidth=1)
        ax1.set_xlim(axis1_lim)
        ax1.set_ylim(axis2_lim)
        ax1.set_xlabel(axis1_name + '-axis')
        ax1.set_ylabel(axis2_name + '-axis')

        ax2.title.set_text('Original Field')
        ax2.quiver(x, y, u_orig_, v_orig_, angles='xy')
        ax2.set_xlim(axis1_lim)
        ax2.set_ylim(axis2_lim)
        ax2.streamplot(x,y,u_orig_,v_orig_,linewidth=1)
        ax2.set_xlabel(axis1_name + '-axis')
        ax2.set_ylabel(axis2_name + '-axis')

        if settings.inplane_error:
            addit = ' in plane'
        else:
            addit = ''

        ax3.title.set_text('error'+ addit)
        ax3.set_xlim(axis1_lim)
        ax3.set_ylim(axis2_lim)
        ax3.set_xlabel(axis1_name + '-axis')
        ax3.set_ylabel(axis2_name + '-axis')
        img1 = ax3.contourf(x, y, error_, 100)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img1, cax=cax, orientation='vertical')

        #axs3.colorbar(img1)
        fig.savefig('..\Output\calculations_'+settings.Name_of_calculation +'\Plots\combined ' + axis + '= ' + (str(height).replace('.', ',')) +'.jpeg', format='jpeg')