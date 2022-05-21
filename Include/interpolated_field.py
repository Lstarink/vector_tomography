# -*- coding: utf-8 -*-
"""
Created on Thu May 12 20:55:56 2022

@author: lstar
"""
import numpy as np


class InterpolatedField:
    def __init__(self, setup, vector_field):
        self.intersections = setup.intersections
        self.vector_field = vector_field
        self.size = setup.intersections.size
        
    def SampleField(self, coordinate):
        u = 0
        v = 0 
        w = 0
        for i in range(self.size):
            distance = ((coordinate[0] - self.intersections.xArray[i])**2 +
                        (coordinate[1] - self.intersections.yArray[i])**2 +
                        (coordinate[2] - self.intersections.zArray[i])**2)**0.5
            
            u +=  self.lambda_vectors[0][i]*distance
            v +=  self.lambda_vectors[1][i]*distance
            w +=  self.lambda_vectors[2][i]*distance
        
        vector = np.array([u,v,w])
        return(vector)
        
    def CalculateLambda(self, scalar_field):
        
        matrix = np.zeros([self.size,self.size])

        for b in range(self.size):
            for a in range(self.size):
                matrix[b][a] = ((self.intersections.xArray[b]-  self.intersections.xArray[a])**2 +  
                                (self.intersections.yArray[b] - self.intersections.yArray[a])**2 + 
                                (self.intersections.zArray[b] - self.intersections.zArray[a])**2)**0.5
        
        lambda_vector =np.linalg.solve(matrix, scalar_field)
        return(lambda_vector)

    def SetLambdas(self):
        self.lambda_vectors = []

        for i in range(3):
            self.lambda_vectors.append(InterpolatedField.CalculateLambda(self, np.transpose(self.vector_field)[i]))
            
    
    def TestInterpolatedField(self):
        this_works_just_fine = True
        
        interpolated_intersection_field = np.zeros([self.size, 3])
        
        for i in range(self.size):
            interpolated_intersection_field[i] = InterpolatedField.SampleField(self, self.intersections.coordinate_list[i])
            if (np.linalg.norm(interpolated_intersection_field[i] - self.vector_field[i]) > 0.001):
                print(i)
                print('error =' , np.linalg.norm(interpolated_intersection_field[i] - self.vector_field[i]))
                this_works_just_fine = False
                break
        return(this_works_just_fine)
            