# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:57:58 2022

@author: lstar
"""
import numpy as np
import matplotlib.pyplot as plt
import settings
import intersection

class setupIntersections:
    def __init__(self, lines, grid, tubes):
        self.grid = grid
        self.lines = lines
        self.tubes = tubes
        self.xArray = []
        self.yArray = []
        self.zArray = []
        self.coordinate_list = []
        self.size = 0
        self.intersectionMatrix = np.zeros([len(lines), len(lines)])
        self.first_intersection = True
        
    """
    This function checks if two lines in R3 intersect.
    """
    def CalcIntersect(self, line1, line2, i, j):
    
        rounding = 5
        if (abs(np.dot(line1.unit_vector, line2.unit_vector)) < 0.99999):
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
                    
                    if (self.grid.intersection_in_grid(intersectionLocation)):
                        if self.first_intersection:
                            self.first_intersection = False
                            self.coordinate_list.append(intersectionLocation)
                            self.xArray.append(intersectionLocation[0])
                            self.yArray.append(intersectionLocation[1])
                            self.zArray.append(intersectionLocation[2])
                            self.size +=1
                            self.intersectionMatrix[i][j] = 1
                        else:
                            for k in range(len(self.coordinate_list)):
                                if (intersectionLocation == self.coordinate_list[k]).all():
                                    self.intersectionMatrix[i][j] = 1
                                    break
                            else:
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

    def calculate_full_rank(self):
        intersection_instances = []
        for i in range(self.size):
            intersection_ = intersection.Intersection(np.array([self.xArray[i], self.yArray[i], self.zArray[i]]))
            for j in range(len(self.tubes)):
                if (self.tubes[j].StepFunction(intersection_.location)):
                    intersection_.add_tube()
                    intersection_.add_vector(self.tubes[j].line.unit_vector)
            intersection_instances.append(intersection_)
        self.intersection_instances = intersection_instances

    def count_rank(self):
        rank_list = []
        tube_present_list = []

        for i in range(self.size):
            self.intersection_instances[i].determine_rank()
            rank_list.append(self.intersection_instances[i].rank)
            tube_present_list.append(self.intersection_instances[i].tubes_present)

        print(rank_list.count(2), 'intersections of rank 2')
        print(rank_list.count(3), 'intersections of full rank')
        print(tube_present_list.count(2), 'intersections with 2 tubes present')
        print(tube_present_list.count(3), 'intersections with 3 tubes present')
        print(tube_present_list.count(4), 'intersections with 4 tubes present')
        print(tube_present_list.count(5), 'intersections with 5 tubes present')
        more_tubes_present = self.size - tube_present_list.count(2) - tube_present_list.count(3) - tube_present_list.count(4) - tube_present_list.count(5)
        print(more_tubes_present, 'intersections with more than 5 tubes present')

        if settings.use_only_full_rank_intersections:
            full_rank_indices = [i for i, x in enumerate(rank_list) if x == 3]

            full_rank_intersections = np.zeros([3, len(full_rank_indices)])
            full_rank_x = np.zeros([len(full_rank_indices)])
            full_rank_y = np.zeros([len(full_rank_indices)])
            full_rank_z = np.zeros([len(full_rank_indices)])

            index = 0
            for i in full_rank_indices:
                for j in range(3):
                    full_rank_x[index] = self.xArray[i]
                    full_rank_y[index] = self.yArray[i]
                    full_rank_z[index] = self.zArray[i]
                index += 1

            self.xArray = full_rank_x
            self.yArray = full_rank_y
            self.zArray = full_rank_z

    def SaveIntersections(self):
        
        #np.save('intersections.npy', np.array([self.xArray, self.yArray, self.zArray]))
        try:    
            np.save('..\Output\calculations_'+settings.Name_of_calculation +'\Intersections.npy', np.array([self.xArray, self.yArray, self.zArray])) 
        except FileExistsError:
            print("Intersections already exists in this directory")  
                
    def plotIntersections(self):
        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111, projection='3d')
        plt.title('setup')
        ax.set_xlabel('x as [m]')
        ax.set_ylabel('y as [m]')
        ax.set_zlabel('z as [m]')
        plt.grid()
        for i in range(len(self.lines)):
            ax.plot([self.lines[i].A[0],self.lines[i].B[0]],
                     [self.lines[i].A[1],self.lines[i].B[1]], 
                     [self.lines[i].A[2],self.lines[i].B[2]], color = 'b')  
         
        plt.plot(self.xArray, self.yArray, self.zArray, 'o', color = 'black')
        plt.show()
        
    def plot_intersections_2dxy(self):
        #xy
        plt.figure(figsize = (10, 10))
        plt.title('Setup x-y vlak')
        plt.xlabel('x as [m]')
        plt.ylabel('y as [m]')
        plt.grid()
        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[0],self.lines[i].B[0]],
                     [self.lines[i].A[1],self.lines[i].B[1]], color = 'b')
        
        plt.plot(self.xArray, self.yArray, 'o', color = 'black')
        plt.show()
        return()
    
    def plot_intersections_2dxz(self):
        #xy
        plt.figure(figsize = (10, 10))
        plt.title('Setup x-z vlak')
        plt.xlabel('x as [m]')
        plt.ylabel('z as [m]')
        plt.grid()
        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[0],self.lines[i].B[0]],
                     [self.lines[i].A[2],self.lines[i].B[2]], color = 'b')
        
        plt.plot(self.xArray, self.zArray, 'o', color = 'black')
        plt.show()
        return()
    
    def plot_intersections_2dyz(self):
        #xy
        plt.figure(figsize = (10, 10))
        plt.title('Setup y-z vlak')
        plt.xlabel('y as [m]')
        plt.ylabel('z as [m]')
        plt.grid()
        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[1],self.lines[i].B[1]],
                     [self.lines[i].A[2],self.lines[i].B[2]], color = 'b')
        
        plt.plot(self.yArray, self.zArray, 'o', color = 'black')
        plt.show()
        return()

                