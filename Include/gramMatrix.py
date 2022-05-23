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
        value_list = []
        with multiprocessing.Pool() as pool:
            value_list.append(pool.starmap(GramMatrix.calculate_index, index_list))

        print(value_list)
        for i, j, value in value_list[0]:
            self.gram_matrix[i][j] = value

        try:
            np.save('..\Output\calculations_'+settings.Name_of_calculation +'\gramMatrix.npy', self.gram_matrix)
        except FileExistsError:
            print("Gram Matrix already exists in this directory")

    def make_instance_list(self):
        gram_matrix_list = []
        for i in range(self.size):
            gram_matrix_list.append(self)
        return gram_matrix_list

    def enumerate_matrix(self):
        index_list = []
        for i in range(self.size):
            for j in range(self.size):
                index_list.append([self, i,j])
        return index_list

    def calculate_index(self, m ,n):
        if (m != n):
            if (abs(np.dot(self.tubes[m].line.unit_vector, self.tubes[n].line.unit_vector)) < 0.99999):

                tube_intersection = tbi.Tube_intersection(self.tubes[m], self.tubes[n], self.resolution,
                                                          self.intersection_matrix[m][n], self.grid_size)
                volume_intersect = tube_intersection.volume_intersect
                volume_check = tube_intersection.v_check
            else:
                volume_intersect = 0  # This only holds if tubewidth smaller than distance between parallel lines
        else:
            volume_intersect = self.tubes[m].volume

        dotUnitVectors = np.dot(self.tubes[m].line.unit_vector, self.tubes[n].line.unit_vector)
        denominator = self.tubes[m].area * self.tubes[n].area * self.tubes[m].line.length * self.tubes[n].line.length

        #self.gram_matrix[m][n] = volume_intersect * dotUnitVectors / denominator
        return [m, n, volume_intersect * dotUnitVectors / denominator]

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
            
