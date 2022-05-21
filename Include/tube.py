# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:55:00 2022

@author: lstar
"""

import numpy as np
import gridSize as gd

class tube:
    def __init__(self, line, width, grid):
        self.grid = grid
        self.line = line
        self.width = width
        self.area = tube.CalculateArea(self)
        self.volume = tube.CalculateVolume(self)
        self.g = tube.calculateG(self)
        tube.tube_angles(self)
     
    def CalculateArea(self):
        area = self.width**2 #calculation for square tubes to keep things simpler when calculating tube volume intersect
        return(area)
    
    def CalculateVolume(self):
        volume = self.line.length*self.area
        return(volume)
    
    def StepFunction(self,location):
        x_n = location[0]
        y_n = location[1]
        z_n = location[2]
        
        
        #First move starting point to origin
        xOrigin = x_n - self.line.A[0]
        yOrigin = y_n - self.line.A[1]
        zOrigin = z_n - self.line.A[2]
        point = np.array([xOrigin, yOrigin, zOrigin])
        
        #Second, calculate dotproduct with unit vector
        projectionLength = np.dot(point,self.line.unit_vector)
        projection = projectionLength*self.line.unit_vector
        
        #third, calculate distance to point
        distanceVector = point - projection
        distance = (distanceVector[0]**2 + distanceVector[1]**2 + distanceVector[2]**2)**0.5
        
        #Check if point in tube
        
        if (distance < 0.5*self.width) and (self.grid.PointInGrid(location)):
            return(1)
        else:
            return(0)
        
    def calculateG(self):
        g = (1/(self.volume))*self.line.unit_vector
        return(g)
    
    def SetEpsilon(self,epsilon):
        self.epsilon = epsilon
        
    
    def basis_cylinder_along_z(radius,height_z):
        z = np.linspace(0, height_z, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid)
        y_grid = radius*np.sin(theta_grid)
        
        cylinder = np.array([x_grid, y_grid, z_grid])
        return (cylinder)
    
    def tube_angles(self):
        if (self.line.unit_vector[0] != 0):
            theta1 = np.arctan(self.line.unit_vector[1]/
                            self.line.unit_vector[0])
        else:
            theta1 = np.pi/2
        
        if (self.line.unit_vector[0]<0):
            theta1 = theta1 + np.pi
        
        if (self.line.unit_vector[2] != 0):
            phi1 = np.arctan(((self.line.unit_vector[0]**2 +self.line.unit_vector[1]**2)**0.5)/
                              self.line.unit_vector[2])
        else:
            phi1 = np.pi/2
        
        if (self.line.unit_vector[2]<0):
            phi1 = phi1 + np.pi
            
        self.phi = phi1
        self.theta = theta1
    
    def Parametric_tube(self):

        cylinder1 = tube.basis_cylinder_along_z(self.width, self.line.length)
            
                
        cylinder1_rotated_once = tube.rotate_something('y', self.phi, cylinder1)
        cylinder1_rotated_twice = tube.rotate_something('z', self.theta, cylinder1_rotated_once)
        cylinder1_translated = tube.translate_something(self.line.A, cylinder1_rotated_twice)
        
        xc1, yc1, zc1 = cylinder1_translated[0], cylinder1_translated[1], cylinder1_translated[2]        
        
        return(xc1, yc1, zc1)
        

        
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

    def translate_something(vector, item_to_translate):
        shape = item_to_translate.shape
        
        translated_item = np.zeros([shape[0], shape[1], shape[2]])
        
        for i in range(shape[1]):
            for j in range(shape[2]):
               translated_item[0][i][j] = item_to_translate[0][i][j] + vector[0]
               translated_item[1][i][j] = item_to_translate[1][i][j] + vector[1]
               translated_item[2][i][j] = item_to_translate[2][i][j] + vector[2]   
               
        return(translated_item)