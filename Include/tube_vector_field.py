# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:26:05 2022

@author: lstar
"""
import numpy as np

class tube_vector_field:
    def __init__(self, tube, intersections): #removed: , epsilon
        self.tube = tube
        self.intersections = intersections
        self.fieldIntersections = tube_vector_field.CalculateUDeltaIntersections(self)

    
    """
    This function determines if the intersection point of two lines is contained by a tube.
    If this is the case it sets a vector corresponding to the tubes vectorfield at this point.
    Used to plot the intersection vectorfields
    """

    def CalculateUDeltaIntersections(self):
        x = self.intersections[0]
        y = self.intersections[1]
        z = self.intersections[2]
        
        field = np.zeros([self.intersections.size, 3])
        
        epsilon = self.tube.epsilon
        g = self.tube.g #(1/(A*l))*unitvector

        
        for i in range(len(x)):

            stepFunction = self.tube.StepFunction([x[i], y[i],z[i]]) #coordinate in tube?
                
            field[i][0] = g[0]*stepFunction*epsilon
            field[i][1] = g[1]*stepFunction*epsilon
            field[i][2] = g[2]*stepFunction*epsilon
                    
        return(field)