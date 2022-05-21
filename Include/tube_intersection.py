# -*- coding: utf-8 -*-
"""
Created on Mon May  9 19:05:15 2022

@author: lstar
"""
import matplotlib.pyplot as plt
import numpy as np
import setup_intersections

class Tube_intersection:
    def __init__(self, tube1, tube2, resolution):
        self.tube1 = tube1
        self.tube2 = tube2
        self.res = resolution
        self.point_on_tube, self.closest_point = Tube_intersection.ClosestPoint(self)        
        self.intersection_angle = Tube_intersection.CalculateIntersectionAngle(self)
        Tube_intersection.integrate_tube(self)
        self.volume_intersect = 0
        Tube_intersection.VolumeIntersect(self)
           
    def VolumeIntersect(self):
       for i in range(self.res):
           for j in range(self.res):
               for k in range(self.res):
                   dV = self.range_radius[i]*self.dz*self.dr*self.dtheta
                   in_other_tube =self.tube2.StepFunction(np.array([self.integration_space[0][i][j][k],
                                                                    self.integration_space[1][i][j][k],
                                                                    self.integration_space[2][i][j][k]]))
                   self.volume_intersect += dV*in_other_tube

       self.v_check = 0.25*np.pi*2*self.delta_on_axis*self.tube1.width**2
       if (self.v_check < self.volume_intersect):
           raise Exception('grammatrix integration not working correctly')
            
    def VolumeIntersectAnalytical(self, m , n):
    #tubes is a list of tubes, m and n are the indices for the gram matrix
    #Maybe add a fourth option in the future for lines that do not intersect, but the corresponding tubes do intersect
        if (intersections.intersectionMatrix[m][n] == 1):
            vector1 = self.tube1.line.unit_vector
            vector2 = self.tube2.line.unit_vector
            
            cross_product = np.cross(vector1,vector2)
            norm_cross = np.linalg.norm(cross_product) #= sin(theta)
            
            volumeIntersect = self.tubes1.width**3/norm_cross
            return(volumeIntersect)
        else:
                volumeIntersect = 0
        return(volumeIntersect)
    
    def CalculateIntersectionAngle(self):
        theta = np.arcsin(np.linalg.norm(np.cross(self.tube1.line.unit_vector, self.tube2.line.unit_vector)))     
        return (theta)
                                
    def rotate_something(axis, theta, item_to_rotate):
    
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
        
        rotated_item = np.zeros([shape[0], shape[1], shape[2], shape[3]])
        
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    vector = np.array([item_to_rotate[0][i][j][k],
                                       item_to_rotate[1][i][j][k],
                                       item_to_rotate[2][i][j][k]])
                    rotated_vector = np.matmul(rotation_matrix, vector)
                    
                    rotated_item[0][i][j][k] = rotated_vector[0]
                    rotated_item[1][i][j][k] = rotated_vector[1]
                    rotated_item[2][i][j][k] = rotated_vector[2]
               
        return(rotated_item)

    def translate_something(vector, item_to_translate):
        shape = item_to_translate.shape
        
        translated_item = np.zeros([shape[0], shape[1], shape[2], shape[3]])
        
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                   translated_item[0][i][j][k] = item_to_translate[0][i][j][k] + vector[0]
                   translated_item[1][i][j][k] = item_to_translate[1][i][j][k] + vector[1]
                   translated_item[2][i][j][k] = item_to_translate[2][i][j][k] + vector[2]   
               
        return(translated_item)    

    def ClosestPoint(self):
        line1 = self.tube1.line
        line2 = self.tube2.line
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
        
        return(intersectionLocation1, closest_point)
    
    def integrate_tube(self):
        
        res = self.res
        if (np.abs(self.intersection_angle) > np.pi/2):     
            intersection_angle = np.pi - self.intersection_angle
        else:
            intersection_angle = self.intersection_angle
            

        delta_on_axis = 0.5*self.tube1.width/np.cos(np.pi/2 - intersection_angle)
        range_theta = np.linspace(0, 2*np.pi, res)
        range_on_axis = np.linspace(-delta_on_axis, delta_on_axis, res)
        range_radius = np.linspace(0, self.tube1.width/2, res)
        
        ##To cartesisian
        all_coordinates = np.zeros([3, res, res, res])
        
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    all_coordinates[0][i][j][k] = np.cos(range_theta[j])*range_radius[i]
                    all_coordinates[1][i][j][k] = np.cos(range_theta[j])*range_radius[i]
                    all_coordinates[2][i][j][k] = range_on_axis[k]
        
        rotated_once = Tube_intersection.rotate_something('y', self.tube1.phi, all_coordinates)
        rotated_twice = Tube_intersection.rotate_something('z', self.tube1.theta, rotated_once)
        translated_once = Tube_intersection.translate_something(self.point_on_tube, rotated_twice)
        
        self.integration_space = translated_once
        
        self.range_radius = range_radius
        self.dz = delta_on_axis*2/res
        self.dr = self.tube1.width/(res*2)
        self.dtheta = 2*np.pi/res
        self.delta_on_axis = delta_on_axis