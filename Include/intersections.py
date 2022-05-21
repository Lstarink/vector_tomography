# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:57:58 2022

@author: lstar
"""
import numpy as np
import matplotlib.pyplot as plt
import settings

class setupIntersections:
    def __init__(self, lines, grid):
        self.grid = grid
        self.lines = lines
        self.xArray = []
        self.yArray = []
        self.zArray = []
        self.intersection_list = []
        self.coordinate_list = []
        self.size = 0
        self.intersectionMatrix = np.zeros([len(lines), len(lines)])
        self.first_intersection = True
        
    """
    This function checks if two lines in R3 intersect.
    """
    def CalcIntersect(self, line1, line2, i, j):
    
        rounding = 5
        #if not(line1.unit_vector == line2.unit_vector).all():#If unit vectors are equal, two lines cannot intersect
        if (abs(np.dot(line1.unit_vector, line2.unit_vector)) < 0.99999):
# =============================================================================
#             print('line1 : ', line1.unit_vector)
#             print('line2 : ', line2.unit_vector)
# =============================================================================
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
            
            dA_check = np.matmul(unitVectorMatrix,locVector)
            intersectionLocation1 = np.array([round((line1.A[0] + line1.unit_vector[0]*locVector[0]),rounding),  
                                             round((line1.A[1] + line1.unit_vector[1]*locVector[0]),rounding),
                                             round((line1.A[2] + line1.unit_vector[2]*locVector[0]),rounding)])
            
            intersectionLocation2 = np.array([round((line2.A[0] + line2.unit_vector[0]*locVector[1]),rounding),  
                                             round((line2.A[1] + line2.unit_vector[1]*locVector[1]),rounding),
                                             round((line2.A[2] + line2.unit_vector[2]*locVector[1]),rounding)])
            
            
            margin = 10**-4
            if (np.linalg.norm(intersectionLocation1 - intersectionLocation2) < margin):
                if (np.linalg.norm(dA_check - dA) < margin):

                    intersectionLocation = intersectionLocation1
                    
                    if (self.grid.PointInGrid(intersectionLocation)):
                        if self.first_intersection:
                            self.first_intersection = False
                            self.intersection_list.append(Intersection(line1, line2, intersectionLocation))
                            self.coordinate_list.append(intersectionLocation)
                            self.xArray.append(intersectionLocation[0])
                            self.yArray.append(intersectionLocation[1])
                            self.zArray.append(intersectionLocation[2])
                            self.size +=1
                            self.intersectionMatrix[i][j] = 1
                        else:
                            for k in range(len(self.coordinate_list)):
                                if (intersectionLocation == self.coordinate_list[k]).all():
                                    self.intersection_list[k].AddLines(line1, line2)
                                    self.intersectionMatrix[i][j] = 1
                                    break
                            else:
                                self.intersection_list.append(Intersection(line1, line2, intersectionLocation))
                                self.coordinate_list.append(intersectionLocation)
                                self.xArray.append(intersectionLocation[0])
                                self.yArray.append(intersectionLocation[1])
                                self.zArray.append(intersectionLocation[2])                                    
                                self.size +=1
                                self.intersectionMatrix[i][j] = 1

    """Sets all intersection points, makes three arrays with the coordinates for all the intersections. Makes
    a nxn matrix that tells if line i and j of the setup intersect"""
    def SetPoints(self):
        for i in range(len(self.lines)):
            for j in range(len(self.lines)):
                setupIntersections.CalcIntersect(self, self.lines[i], self.lines[j], i, j)
    
    def SaveIntersections(self):
        
        #np.save('intersections.npy', np.array([self.xArray, self.yArray, self.zArray]))
        try:    
            np.save('..\Output\calculations_'+settings.Name_of_calculation +'\Intersections.npy', np.array([self.xArray, self.yArray, self.zArray])) 
        except FileExistsError:
            print("Intersections already exists in this directory")  
                
    def plotIntersections(self):
        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(self.lines)):
            ax.plot([self.lines[i].A[0],self.lines[i].B[0]],
                     [self.lines[i].A[1],self.lines[i].B[1]], 
                     [self.lines[i].A[2],self.lines[i].B[2]], color = 'b')  
         
        plt.plot(self.xArray, self.yArray, self.zArray, 'o', color = 'black')
        plt.show()
        
    def plot_intersections_2dxy(self):
        #xy
        plt.figure(figsize = (10, 10))
        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[0],self.lines[i].B[0]],
                     [self.lines[i].A[1],self.lines[i].B[1]], color = 'b')
        
        plt.plot(self.xArray, self.yArray, 'o', color = 'black')
        plt.show()
        return()
    
    def plot_intersections_2dxz(self):
        #xy
        plt.figure(figsize = (10, 10))
        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[0],self.lines[i].B[0]],
                     [self.lines[i].A[2],self.lines[i].B[2]], color = 'b')
        
        plt.plot(self.xArray, self.zArray, 'o', color = 'black')
        plt.show()
        return()
    
    def plot_intersections_2dyz(self):
        #xy
        plt.figure(figsize = (10, 10))
        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[1],self.lines[i].B[1]],
                     [self.lines[i].A[2],self.lines[i].B[2]], color = 'b')
        
        plt.plot(self.yArray, self.zArray, 'o', color = 'black')
        plt.show()
        return()
        
class Intersection:
    def __init__(self, line_1, line_2, intersection_location):
        self.line_1 = line_1
        self.line_2 = line_2
        self.intersection_location = intersection_location
        self.rank = 2
        self.line_3 = None
        
    def AddLines(self, line_3, line_4):
        if (line_3 != self.line_1) and (line_3 != self.line_2):
            self.line_3 = line_3
            self.rank+=1
        if (line_4 != self.line_1) and (line_4 != self.line_2):
            if (self.line_3 == None):
                self.line_3 = line_4
            else: 
                self.line_4 = line_4
                self.rank += 1
                