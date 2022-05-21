# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:50:38 2022

@author: lstar
"""

class line_group:
    def __init__(self, unit_vector):
        self.unit_vector = unit_vector
        self.lines_k = []
        self.group_size = int(0)
        self.v_average_k = float()

        
    def AddLine(self, line):
        self.lines_k.append(line)
        self.group_size += 1
    
    def calculateVK(self):
        vSum = 0
        for i in range(self.group_size):
            vSum += self.lines_k[i].v_average
        
        self.v_average_k = vSum/self.group_size