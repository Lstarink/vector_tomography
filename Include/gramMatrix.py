# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:03:25 2022

@author: lstar
"""
import numpy as np
import matplotlib.pyplot as plt
import tube_intersection as tbi
import multiprocessing
import time
import sys

#sys.path.append('C:/Users/lstar/Documents/TUDelft/Jaar3/BEP/Python/3D/Mark2/Settings')
import settings

class GramMatrix:
    def __init__(self, tubes, grid_size, intersection_matrix):
        self.intersection_matrix = intersection_matrix
        self.grid_size = grid_size
        self.tubes = tubes
        self.size = len(tubes)
        self.gram_matrix = np.zeros([self.size, self.size])
        self.resolution = settings.matrix_integration_setting
        if settings.use_integration_for_gram_matrix:
            self.first_element = True
            GramMatrix.MakeGramMatrix_integrated(self)
            time.sleep(10)
            print(self.gram_matrix)
        else:
            GramMatrix.MakeGramMatrix_analytical(self)

    def MakeGramMatrix_integrated(self):
        index_list = GramMatrix.enumerate_matrix(self)
        gram_matrix_list = GramMatrix.make_instance_list(self)
        with multiprocessing.Pool() as pool:
            pool.starmap(GramMatrix.calculate_index, index_list)

        try:
            np.save('..\Output\calculations_'+settings.Name_of_calculation +'\gramMatrix.npy', self.gram_matrix)
        except FileExistsError:
            print("Gram Matrix already exists in this directory")

    def make_instance_list(self):
        gram_matrix_list = []
        for i in range(self.size):
            gram_matrix_list.append(self)
        return(gram_matrix_list)

    def enumerate_matrix(self):
        index_list = []
        for i in range(self.size):
            for j in range(self.size):
                index_list.append([self, i,j])
        return(index_list)

    def calculate_index(self, m ,n):
        if (m != n):
            if (abs(np.dot(self.tubes[m].line.unit_vector, self.tubes[n].line.unit_vector)) < 0.99999):

                tube_intersection = tbi.Tube_intersection(self.tubes[m], self.tubes[n], self.resolution,
                                                          self.intersection_matrix[m][n], self.grid_size)
                volume_intersect = tube_intersection.volume_intersect
                volume_check = tube_intersection.v_check
                if settings.plot_tube_intersections and (
                        volume_intersect != volume_check):  # only plot nontrivial cases
                    GramMatrix.PlotTwoTubes(self, self.tubes[m], self.tubes[n], volume_intersect, volume_check)
            else:
                volume_intersect = 0  # This only holds if tubewidth smaller than distance between parallel lines
        else:
            volume_intersect = self.tubes[m].volume

        dotUnitVectors = np.dot(self.tubes[m].line.unit_vector, self.tubes[n].line.unit_vector)
        denominator = self.tubes[m].area * self.tubes[n].area * self.tubes[m].line.length * self.tubes[n].line.length

        self.gram_matrix[m][n] = volume_intersect * dotUnitVectors / denominator
        #print(volume_intersect * dotUnitVectors / denominator)

    def MakeGramMatrix_analytical(self):
        for m in range(self.size):
            for n in range(self.size):
                if (m != n):
                    if (abs(np.dot(self.tubes[m].line.unit_vector, self.tubes[n].line.unit_vector)) < 0.99999):
                        volume_intersect = GramMatrix.TubesVolumeIntersectAnalytical(self.tubes, m, n, self.intersection_matrix)
                    else:
                        volume_intersect = 0 #This only holds if tubewidth smaller than distance between parallel lines
                else:
                    volume_intersect = self.tubes[m].volume
                dotUnitVectors = np.dot(self.tubes[m].line.unit_vector, self.tubes[n].line.unit_vector)
                denominator = self.tubes[m].area*self.tubes[n].area*self.tubes[m].line.length*self.tubes[n].line.length
                self.gram_matrix[m][n] = volume_intersect*dotUnitVectors/denominator
        try:
            np.save('..\Output\calculations_'+settings.Name_of_calculation +'\gramMatrix.npy', self.gram_matrix) 
        except FileExistsError:
            print("Gram Matrix already exists in this directory")  

    def TubesVolumeIntersectAnalytical(tubes, m, n, intersectionMatrix):
        #tubes is a list of tubes, m and n are the indices for the gram matrix
        #Maybe add a fourth option in the future for lines that do not intersect, but the corresponding tubes do intersect
        if (m == n):
            volumeIntersect = tubes[m].volume
            return(volumeIntersect)
        else:
            if (intersectionMatrix[m][n] == 1):
                vector1 = tubes[m].line.unit_vector
                vector2 = tubes[n].line.unit_vector
                
                cross_product = np.cross(vector1,vector2)
                norm_cross = np.linalg.norm(cross_product) #= sin(theta)
                
                volumeIntersect = tubes[m].width**3/norm_cross
                return(volumeIntersect)
            else:
                volumeIntersect = 0
            return(volumeIntersect)        
            
    def basis_cylinder_along_z(radius,height_z):
        z = np.linspace(0, height_z, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid)
        y_grid = radius*np.sin(theta_grid)
        
        cylinder = np.array([x_grid, y_grid, z_grid])
        return (cylinder)
    
    def PlotTwoTubes(self, tube1, tube2, v, v_check):
        line1 = tube1.line
        line2 = tube2.line
        cylinder1 = GramMatrix.basis_cylinder_along_z(tube1.width, tube1.line.length)
        cylinder2 = GramMatrix.basis_cylinder_along_z(tube2.width, tube2.line.length)
            
        phi1 = tube1.phi
        theta1 = tube1.theta

        phi2 = tube2.phi
        theta2 = tube2.theta
        
        cylinder1_rotated_once = GramMatrix.rotate_something(self, 'y', phi1, cylinder1)
        cylinder1_rotated_twice = GramMatrix.rotate_something(self, 'z', theta1, cylinder1_rotated_once)
        cylinder1_translated = GramMatrix.translate_something(self, tube1.line.A, cylinder1_rotated_twice)
        
        cylinder2_rotated_once = GramMatrix.rotate_something(self, 'y', phi2, cylinder2)
        cylinder2_rotated_twice = GramMatrix.rotate_something(self, 'z', theta2, cylinder2_rotated_once)
        cylinder2_translated = GramMatrix.translate_something(self, tube2.line.A, cylinder2_rotated_twice)
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        plt.title('volume intersect ='+  str(v) + 'volume check =' + str(v_check))
        plt.xlim(self.grid_size.x_min, self.grid_size.x_max)
        plt.ylim(self.grid_size.y_min, self.grid_size.y_max)
        ax.set_zlim(self.grid_size.z_min, self.grid_size.z_max)
        lines = [line1, line2]
        for i in range(len(lines)):
            ax.plot([lines[i].A[0],lines[i].B[0]],
                     [lines[i].A[1],lines[i].B[1]], 
                     [lines[i].A[2],lines[i].B[2]], color = 'b')  
        xc1, yc1, zc1 = cylinder1_translated[0], cylinder1_translated[1], cylinder1_translated[2]
        xc2, yc2, zc2 = cylinder2_translated[0], cylinder2_translated[1], cylinder2_translated[2]
        ax.plot_surface(xc1, yc1, zc1, alpha = 0.5)        
        ax.plot_surface(xc2, yc2, zc2, alpha = 0.5)

        plt.show()
        return()
                
    def rotate_something(self, axis, theta, item_to_rotate):
    
        if (axis == 'x'):
            rotation_matrix = np.array([[1,  0,  0],
                                       [0, np.cos(theta), -np.sin(theta)],
                                       [0, np.sin(theta), np.cos(theta)]])
        elif (axis == 'y'):
            rotation_matrix = np.array([[np.cos(theta),  0,  np.sin(theta)],
                                        [0, 1, 0],
                                        [-np.sin(theta), 0, np.cos(theta)]])
        elif (axis == 'z'):
            rotation_matrix = np.array([[np.cos(theta),  -np.sin(theta),  0],
                                        [np.sin(theta),  np.cos(theta), 0],
                                        [0, 0, 1]])    
        else:
            raise ValueError('axis not an axis')
        
        shape = item_to_rotate.shape
        
        rotated_item = np.zeros([shape[0], shape[1], shape[2]])
        
        for i in range(shape[1]):
            for j in range(shape[2]):
               vector = np.array([item_to_rotate[0][i][j],
                                  item_to_rotate[1][i][j],
                                  item_to_rotate[2][i][j]])
               rotated_vector = np.matmul(rotation_matrix, vector)
               
               rotated_item[0][i][j] = rotated_vector[0]
               rotated_item[1][i][j] = rotated_vector[1]
               rotated_item[2][i][j] = rotated_vector[2]
               
        return(rotated_item)

    def translate_something(self, vector, item_to_translate):
        shape = item_to_translate.shape
        
        translated_item = np.zeros([shape[0], shape[1], shape[2]])
        
        for i in range(shape[1]):
            for j in range(shape[2]):
               translated_item[0][i][j] = item_to_translate[0][i][j] + vector[0]
               translated_item[1][i][j] = item_to_translate[1][i][j] + vector[1]
               translated_item[2][i][j] = item_to_translate[2][i][j] + vector[2]   
               
        return(translated_item)    

    def ClosestPoint(self, line1, line2):
                #find point closest to both lines
        rounding = 5
        dA = np.array([line1.A[0] - line2.A[0], 
                       line1.A[1] - line2.A[1],
                       line1.A[2] - line2.A[2]])
        
        unitVectorMatrix = np.array([[-line1.unit_vector[0], line2.unit_vector[0]],
                                    [-line1.unit_vector[1], line2.unit_vector[1]],
                                    [-line1.unit_vector[2], line2.unit_vector[2]]])
        
        #locVector = np.linalg.solve(unitVectorMatrix,dA)
        B__2 = np.matmul(np.transpose(unitVectorMatrix),unitVectorMatrix)
        B__2_inverse = np.linalg.inv(B__2)
        B__3 = np.matmul(B__2_inverse,np.transpose(unitVectorMatrix))
        locVector = np.matmul(B__3,dA)
        intersectionLocation1 = np.array([round((line1.A[0] + line1.unit_vector[0]*locVector[0]),rounding),  
                                          round((line1.A[1] + line1.unit_vector[1]*locVector[0]),rounding),
                                          round((line1.A[2] + line1.unit_vector[2]*locVector[0]),rounding)])
        
        intersectionLocation2 = np.array([round((line2.A[0] + line2.unit_vector[0]*locVector[1]),rounding),  
                                          round((line2.A[1] + line2.unit_vector[1]*locVector[1]),rounding),
                                          round((line2.A[2] + line2.unit_vector[2]*locVector[1]),rounding)])
                
        closest_point = (intersectionLocation1+intersectionLocation2)/2
        
        if False:
            fig = plt.figure(figsize=(10,10))
            lines = [line1, line2]
            ax = fig.add_subplot(111, projection='3d')
            point3 = closest_point
            for i in range(len(lines)):
                ax.plot([lines[i].A[0],lines[i].B[0]],
                         [lines[i].A[1],lines[i].B[1]], 
                         [lines[i].A[2],lines[i].B[2]], color = 'b')  
             
            x = [intersectionLocation1[0], intersectionLocation2[0], point3[0]]
            y = [intersectionLocation1[1], intersectionLocation2[1], point3[1]]
            z = [intersectionLocation1[2], intersectionLocation2[2], point3[2]]
            plt.plot(x,y,z, 'o', color = 'black')
            plt.show()
        return(closest_point)
    
def main():
    import line as ln
    import tube as tb
    import gridSize as grid
    import setup_intersections as ints
    
    width = 0.01
    
    A1 = np.array([0, 0, 0])
    B1 = np.array([0, 0, 1])
    line1 = ln.line(A1, B1)
    tube1 = tb.tube(line1, width)
    
    A2 = np.array([-0.5, 0, 0.5])
    B2 = np.array([0.5, 0, 0.5])
    line2 = ln.line(A2, B2)
    tube2 = tb.tube(line2, width)
    
    lines = [line1, line2]
    tubes = [tube1, tube2]
    
    grid_size = grid.gridSize(lines)
    intersections = ints.setupIntersections(lines, grid)
    
    gram_matrix = GramMatrix(tubes, grid_size, intersections.intersectionMatrix)    
    
    gram_matrix.integrate_tube(0, np.pi, tube1)
    return(0)

if __name__ == "__main__":
    main()
    
    
    
    
    
    