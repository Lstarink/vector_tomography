# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:31:15 2022

@author: lstar
"""

import numpy as np
import settings

class gridSize:
    def __init__(self, lines, offset=None):
        self.offset = offset
        gridSize.CalculateRange(self, lines)

        
    def CalculateRange(self, lines):
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        z_min = 0
        z_max = 0
                
        for i in range(len(lines)):
            #set minima x
            if (lines[i].A[0] < x_min):
                x_min = lines[i].A[0]
            if (lines[i].B[0] < x_min):
                x_min = lines[i].B[0]
            #set minima y
            if (lines[i].A[1] < y_min):
                y_min = lines[i].A[1]
            if (lines[i].B[1] < y_min):
                y_min = lines[i].B[1]
            #set minima z
            if (lines[i].A[2] < z_min):
                z_min = lines[i].A[2]
            if (lines[i].B[2] < z_min):
                z_min = lines[i].B[2]
                
            #set maxima x
            if (lines[i].A[0] > x_max):
                x_max = lines[i].A[0]
            if (lines[i].B[0] > x_max):
                x_max = lines[i].B[0]
            #set maxima y
            if (lines[i].A[1] > y_max):
                y_max = lines[i].A[1]
            if (lines[i].B[1] > y_max):
                y_max = lines[i].B[1]
            #set maxima z
            if (lines[i].A[2] > z_max):
                z_max = lines[i].A[2]
            if (lines[i].B[2] > z_max):
                z_max = lines[i].B[2]
        
        self.size = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    
    
    def PointInGrid(self, point):
        if (point[0] >= self.size[0]) and (point[0] <= self.size[3]):
                if (point[1] >= self.size[1]) and (point[1] <= self.size[4]):
                    if (point[2] >= self.size[2]) and (point[2] <= self.size[5]):
                        return True
                    
        else:
            return False
                            
    def intersection_in_grid(self, point):
        boundary_edge_x = settings.intersection_boundary_edge_x
        boundary_edge_y = settings.intersection_boundary_edge_y
        boundary_edge_z = settings.intersection_boundary_edge_z
        if (point[0] >= self.size[0] + boundary_edge_x) and (point[0] <= self.size[3] - boundary_edge_x):
            if (point[1] >= self.size[1] + boundary_edge_y) and (point[1] <= self.size[4] - boundary_edge_y):
                if (point[2] >= self.size[2] + boundary_edge_z) and (point[2] <= self.size[5] - boundary_edge_z):
                    return True

        else:
            return False
