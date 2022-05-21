# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:48:51 2022

@author: lstar
"""
import numpy as np
import settings

class line:
    def __init__(self, A, B):
        self.A = A                  #starting point of the line
        self.B = B                  #end point of the line
        self.length = line.Length(self)
        self.unit_vector = line.Unit_vector(self)
    
    "returns unit vector of the line"
    def Unit_vector(self):
        unit_vector = np.array([self.B[0]-self.A[0],
                                self.B[1]-self.A[1],
                                self.B[2]-self.A[2]])/self.length
        return(unit_vector)
    
    "returns length of the line"   
    def Length(self):
        dX = self.B[0] - self.A[0]
        dY = self.B[1] - self.A[1]
        dZ = self.B[2] - self.A[2]
        
        length = (dX**2 + dY**2 + dZ**2)**0.5
        return(length)
    
    "setter for v_average"
    def Set_V_Average(self, v_average):
        self.v_average = v_average
        
    "Calculates the difference between the line integral over U0 and the measured value"    
    def CalculateVDelta(self, field):
        #This function performs a line integral over the line in a given field
        iteration_steps = settings.line_integral_iteration_steps               #Numerical integration setting
        ds = self.length/iteration_steps
        
        v_sum = 0
        
        for i in range(0, iteration_steps):
            location_x = self.A[0] + self.unit_vector[0]*ds*i
            location_y = self.A[1] + self.unit_vector[1]*ds*i
            location_z = self.A[2] + self.unit_vector[2]*ds*i
            
            v = field.Sample(location_x, location_y, location_z) #U(x,y,z)
            
            v_projected = np.dot(v,self.unit_vector)
            v_sum += v_projected
            
        v_average_integrated = v_sum/iteration_steps
        
        self.vDelta = self.v_average - v_average_integrated