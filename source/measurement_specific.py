# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:43:58 2022

@author: lstar
"""
import numpy as np
import matplotlib.pyplot as plt
import settings
import sympy as sp
import vector_field
import tube_vector_field as tvf
import interpolated_field as interp
import intersection
import error

class Measurement:
    def __init__(self, setup):
        self.setup = setup
        self.measurements = np.load('../Measurements/' + settings.measurement_file)
        self.gram_matrix = np.load('../Output/' +'calculations_'+ settings.Name_of_calculation + '/gramMatrix.npy')
        self.intersections = np.load('../Output/' + 'calculations_'+ settings.Name_of_calculation + '/Intersections.npy')
     
    
    def AddVToLine(self):
        for i in range(len(self.setup.lines)):
            try:    
                self.setup.lines[i].Set_V_Average(self.measurements[i])
            except IndexError:
                print('you probably tried to calculate the measurements from one setup with a different setup')
    
    def CalculateVk(self):
        for i in range(self.setup.number_of_linegroups):
            self.setup.lineGroups[i].calculateVK()
    
    def CalculateU0(self):
        lineGroups = self.setup.lineGroups
        B = []
        V = []
        for i in range(len(lineGroups)):
            B.append(np.array(lineGroups[i].unit_vector, dtype=float))
            V.append(lineGroups[i].v_average_k)
    
        B__2 = np.matmul(np.transpose(B),B)
        B__2_inverse = np.linalg.inv(B__2)
        B__3 = np.matmul(B__2_inverse,np.transpose(B))
        betas = np.matmul(B__3,V)
        
        self.u0 = vector_field.vector_field(sp.Float(betas[0]), sp.Float(betas[1]), sp.Float(betas[2]))
        print('u0 = ', betas)
        return()
                
    def Plot_U0(self):
        self.u0.Plot(self.setup.grid, 7)
        
    def CalculateVDelta_(self):
        for i in range(len(self.setup.lines)):
             self.setup.lines[i].CalculateVDelta(self.u0)   
    
    def SolveGramMatrix(self):
        vector_v_delta = np.zeros([len(self.setup.lines)])
        #print(self.gram_matrix)
        for i in range(len(self.setup.lines)):
            vector_v_delta[i] = self.setup.lines[i].vDelta
            
        self.vector_epsilon = np.linalg.solve(self.gram_matrix, vector_v_delta)
        
        for i in range(len(self.vector_epsilon)):
            self.setup.tubes[i].SetEpsilon(self.vector_epsilon[i])
            
    def MakeTubeVectorFields(self):
        self.UDeltaList = []
        for i in range(len(self.setup.tubes)):
            self.UDeltaList.append(tvf.tube_vector_field(self.setup.tubes[i], self.intersections))
                
    def castU0Intersections(self):
        x = self.intersections[0]
        y = self.intersections[0]
        z = self.intersections[0]
    
        self.u0Casted = np.zeros([len(x), 3])
        
        for i in range(len(x)):
            self.u0Casted[i][:] = self.u0.Sample(x[i],y[i],z[i])

    def make_intersection_instances(self):
        self.intersection_instances = []
        for i in range(len(self.intersections[0])):
            self.intersection_instances.append(intersection.Intersection([self.intersections[0][i],
                                                                         self.intersections[1][i],
                                                                         self.intersections[2][i]]))
    def addFieldsIntersections(self, field1, field2, size):
        fieldCombined = np.zeros([size, 3])
        for i in range(size):
            for k in range(3):
                fieldCombined[i][k] = field1[i][k] + field2[i][k]
        return(fieldCombined)
    
    def addFields(self):
        fieldsum = self.u0Casted
        
        for i in range(len(self.UDeltaList)):
            fieldsum = Measurement.addFieldsIntersections(self, fieldsum, self.UDeltaList[i].fieldIntersections, len(self.intersections[0]))
        
        self.final_field = fieldsum

    def plotIntersectionField(self):
        x = self.intersections[0]
        y = self.intersections[1]
        z = self.intersections[2]
        
        field = self.final_field
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, field[:,0], field[:,1], field[:,2])
        plt.title('Reconstructed field sampled at intersections')
        ax.set_xlabel('x as [m]')
        ax.set_ylabel('y as [m]')
        ax.set_zlabel('z as [m]')
        ax.plot(x,y,z,'o', color = 'black')
        for i in range(len(self.setup.lines)):
            ax.plot([self.setup.lines[i].A[0],self.setup.lines[i].B[0]],
                     [self.setup.lines[i].A[1],self.setup.lines[i].B[1]],
                     [self.setup.lines[i].A[2],self.setup.lines[i].B[2]], color = 'b', linestyle = '--', linewidth = .5)
        
        ax.set_xlim([self.setup.grid.x_min, self.setup.grid.x_max])
        ax.set_ylim([self.setup.grid.y_min, self.setup.grid.y_max])
        ax.set_zlim([self.setup.grid.z_min, self.setup.grid.z_max])
        plt.show()
    
    def PlotError(self, generated_field):
        error = self.final_field - generated_field
        
        X = self.intersections[0]
        Y = self.intersections[1]
        Z = self.intersections[2]
        #print(len(error))
        error_absolute = np.zeros([len(error)])
        for i in range(len(error)):
            error_absolute[i] = np.linalg.norm(error[i])/(np.linalg.norm(generated_field[i]))
        
        fig = plt.figure(figsize=(15,15))
        ax = plt.axes(projection="3d")
        plt.title('relative magnitude of the error')
        # Creating plot
        img = ax.scatter3D(X, Y, Z, c =error_absolute, alpha=0.7, marker='.')
        fig.colorbar(img)
        plt.show()
         
    def Interpolate(self):
        self.interpolated_field = interp.InterpolatedField(self.setup, self.final_field, self.intersections)
        self.interpolated_field.SetLambdas()
        print('Interpolation working properly is', self.interpolated_field.TestInterpolatedField())
        
    def PlotInterPolated(self):
        grid = self.setup.grid
        
        res = settings.plot_interpolated_resolution
        
        x = np.linspace(grid.x_min, grid.x_max, res)
        y = np.linspace(grid.y_min, grid.y_max, res)
        z = np.linspace(grid.z_min, grid.z_max, res)
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        u = np.zeros([res, res ,res])
        v = np.zeros([res, res, res])
        w = np.zeros([res, res, res])
        
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    vector = self.interpolated_field.SampleField(np.array([x[i], y[j], z[k]]))
                    u[i][j][k] = vector[0]
                    v[i][j][k] = vector[1]
                    w[i][j][k] = vector[2]
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Interpolated field')
        ax.set_xlabel('x as [m]')
        ax.set_ylabel('y as [m]')
        ax.set_zlabel('z as [m]')
        ax.quiver(X, Y, Z, u, v, w)
        plt.show()

    def plot_field_slices(self, vector_field=None):
        x = np.linspace(self.setup.grid.x_min, self.setup.grid.x_max, settings.plot_amount_of_interpolated_slices)
        y = np.linspace(self.setup.grid.y_min, self.setup.grid.y_max, settings.plot_amount_of_interpolated_slices)
        z = np.linspace(self.setup.grid.z_min, self.setup.grid.z_max, settings.plot_amount_of_interpolated_slices)

        error_ = error.Error(vector_field, self.interpolated_field, self.setup.grid, self.intersections)

        for i in range(settings.plot_amount_of_interpolated_slices):
            error_.SliceZ(z[i])

        for i in range(settings.plot_amount_of_interpolated_slices):
            error_.SliceY(y[i])

        for i in range(settings.plot_amount_of_interpolated_slices):
            error_.SliceX(x[i])

    def calculate_global_error(self, vector_field):
        error_ = error.Error(vector_field, self.interpolated_field, self.setup.grid, self.intersections)
        error_.Intersection_error()
        error_.Global_Error()

def  make_measurement_calculation(setup, generated_field=None, vector_field=None):
    print('loading files...')
    measurement = Measurement(setup)
    print('adding measurements to lines...')
    measurement.AddVToLine()
    measurement.CalculateVk()
    print('calculating u0...')
    measurement.CalculateU0()
    if settings.plot_u0:
        measurement.Plot_U0()
    print('calculating v delta')
    measurement.CalculateVDelta_()
    print('solving Gram Matrix...')
    measurement.SolveGramMatrix()
    print('Making tube_vector_fields')
    measurement.MakeTubeVectorFields()
    measurement.make_intersection_instances()
    print('CastU0')
    measurement.castU0Intersections()
    print('adding everything together...')
    measurement.addFields()
    if settings.plot_intersection_field:
        measurement.plotIntersectionField()
    print('interpolating')
    measurement.Interpolate()
    if settings.plot_interpolated:
        print('sampeling and plotting interpolated field...')
        measurement.PlotInterPolated()
    print('calculating error...')
    if settings.plot_error:
        measurement.PlotError(generated_field)
    if settings.plot_field_sliced:
        measurement.plot_field_slices(vector_field)
    if settings.calculate_error:
        measurement.calculate_global_error(vector_field)
    return(measurement.final_field)

