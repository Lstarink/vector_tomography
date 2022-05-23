# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:56:11 2022

@author: lstar
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

import os
import settings
import line
import line_group
import setup_intersections as ins
import gridSize as grid
import tube as tb
import gramMatrix as gm

class Measurement_setup:
    def __init__(self, file):
        self.file_name = file
        self.lines = Measurement_setup.MakeLines(self)
        self.grid = grid.gridSize(self.lines)
        

    "Loops through the rows of the input file, constructs a line for every row of the file, and appends the line to lines"
    def MakeLines(self):
        lines = []
        line_count = 0
        
        with open(self.file_name) as setup:
            csv_reader = csv.reader(setup, delimiter=',')
            for row in csv_reader:
                lines.append(line.line([float(row[1]),float(row[2]),float(row[3])],
                                  [float(row[4]),float(row[5]),float(row[6])]))
    
                line_count += 1
            
            self.line_count = line_count
        return(lines)
           
    def plot_setup(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.title('setup')
        ax.set_xlabel('x as [m]')
        ax.set_ylabel('y as [m]')
        ax.set_zlabel('z as [m]')

        for i in range(len(self.lines)):
            ax.plot([self.lines[i].A[0], self.lines[i].B[0]],
                     [self.lines[i].A[1], self.lines[i].B[1]],
                     [self.lines[i].A[2], self.lines[i].B[2]], color='b')
        plt.show()
        return()
    
    def plot_setup_2dxy(self):
        #xy
        plt.figure(figsize=(10, 10))
        plt.title('Setup x-y vlak')
        plt.xlabel('x as [m]')
        plt.ylabel('y as [m]')

        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[0], self.lines[i].B[0]],
                     [self.lines[i].A[1], self.lines[i].B[1]], color='b')
        plt.show()
        return()
    
    def plot_setup_2dxz(self):
        #xy
        plt.figure(figsize=(10, 10))
        plt.title('Setup x-z vlak')
        plt.xlabel('x as [m]')
        plt.ylabel('z as [m]')

        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[0], self.lines[i].B[0]],
                     [self.lines[i].A[2], self.lines[i].B[2]], color='b')
        plt.show()
        return()
    
    def plot_setup_2dyz(self):
        #xy
        plt.figure(figsize=(10, 10))
        plt.title('Setup y-z vlak')
        plt.xlabel('y as [m]')
        plt.ylabel('z as [m]')

        for i in range(len(self.lines)):
            plt.plot([self.lines[i].A[1], self.lines[i].B[1]],
                     [self.lines[i].A[2], self.lines[i].B[2]], color='b')
        plt.show()
        return()
        
    
    """
    This function loops through all lines and checks how many different unit vectors the setup has.
    For every new unit vector it makes a new line group and adds the line to that line group.
    If a line has a unit vector for which a line group already exists, it appends this line to that line group 
    """
    def MakeLineGroups(self):
    
        list_unit_vectors = []
        self.number_of_linegroups = 0
        self.lineGroups = []
        first_time = True


        for i in range(self.line_count):
            unit_vector_present = False

            if (first_time):
                first_time = False
                list_unit_vectors.append(self.lines[i].unit_vector)
                new_line_group = line_group.line_group(self.lines[i].unit_vector)
                new_line_group.AddLine(self.lines[i])
                self.lineGroups.append(new_line_group)
                self.number_of_linegroups += 1
            else:
                for j in range(self.number_of_linegroups):
                    if (np.dot(self.lines[i].unit_vector, list_unit_vectors[j]) >0.9999):

                        unit_vector_present = True
                        index = j
                        break
     
                if unit_vector_present:
                    self.lineGroups[index].AddLine(self.lines[i])
                else:
                    list_unit_vectors.append(self.lines[i].unit_vector)
                    new_line_group = line_group.line_group(self.lines[i].unit_vector)
                    new_line_group.AddLine(self.lines[i])
                    self.lineGroups.append(new_line_group)
                    self.number_of_linegroups += 1
        
        self.list_unit_vectors = list_unit_vectors
        
        """TEST"""
        total_lines = 0
        for i in range(self.number_of_linegroups):
            total_lines += self.lineGroups[i].group_size
        if (total_lines == self.line_count):
            print("line groups constructed succesfully")
        else:
            raise Exception('Line groups constructed unsuccesfully')
        
    def SetIntersections(self):
        self.intersections = ins.setupIntersections(self.lines, self.grid, self.tubes)
        self.intersections.SetPoints()
        print('amount of intersections =', len(self.intersections.xArray))
        self.intersections.calculate_full_rank()
        self.intersections.count_rank()
        self.intersections.SaveIntersections()
        print('amount of used intersections =', len(self.intersections.xArray))
        
    def MakeTubes(self, width):
        self.tubes = []
        for i in range(self.line_count):
           self.tubes.append(tb.tube(self.lines[i],width, self.grid))

        if settings.plot_tube_setup:
            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111, projection='3d')
            plt.title('setup')
            ax.set_xlabel('x as [m]')
            ax.set_ylabel('y as [m]')
            ax.set_zlabel('z as [m]')
            plt.xlim(self.grid.size[0], self.grid.size[3])
            plt.ylim(self.grid.size[1], self.grid.size[4])
            ax.set_zlim(self.grid.size[2], self.grid.size[5])

            for k in range(self.line_count):
                xc1, yc1, zc1 = self.tubes[k].Parametric_tube()
                ax.plot_surface(xc1, yc1, zc1, alpha=0.5, color='b')

            plt.show()

        if settings.plot_tube_setup_2d:

            plt.figure(figsize=(15,15))
            plt.title('Setup x-y vlak')
            plt.xlabel('x as [m]')
            plt.ylabel('y as [m]')
            for k in range(self.line_count):
                xc1, yc1, zc1 = self.tubes[k].Parametric_tube()
                plt.plot(xc1, yc1, color='b')
            plt.show()

            plt.figure(figsize=(15,15))
            plt.title('Setup x-z vlak')
            plt.xlabel('x as [m]')
            plt.ylabel('z as [m]')
            for k in range(self.line_count):
                xc1, yc1, zc1 = self.tubes[k].Parametric_tube()
                plt.plot(xc1, zc1, color='b')
            plt.show()

            plt.figure(figsize=(15,15))
            plt.title('Setup y-z vlak')
            plt.xlabel('y as [m]')
            plt.ylabel('z as [m]')
            for k in range(self.line_count):
                xc1, yc1, zc1 = self.tubes[k].Parametric_tube()
                plt.plot(yc1, zc1, color='b')
            plt.show()




    def MakeGramMatrix(self):
        self.gram_matrix = gm.GramMatrix(self.tubes, self.grid, self.intersections.intersectionMatrix)
        
    def MakeDirectory(self):
        file_save_dir = '..\Output\calculations_' + settings.Name_of_calculation   
        try:
            os.mkdir(file_save_dir)
            print("Directory ", file_save_dir,  " Created ")
        except FileExistsError:
            print("Directory ", file_save_dir,  " already exists")


def Make_Setup(filename):

    print('Loading data file...')
    setup = Measurement_setup(filename)

    print(setup.line_count, 'lines constructed')
    if settings.plot_line_setup:
        setup.plot_setup()
        setup.plot_setup_2dxy()
        setup.plot_setup_2dxz()
        setup.plot_setup_2dyz()
    print('Grouping lines together...')
    setup.MakeLineGroups()
    print('making tubes')
    setup.MakeTubes(settings.tube_width)
    print('Making directory for output files')
    setup.MakeDirectory()
    print('calculating intersections')
    setup.SetIntersections()
    if settings.plot_line_intersections:
        setup.intersections.plotIntersections()
        setup.intersections.plot_intersections_2dxy()
        setup.intersections.plot_intersections_2dxz()
        setup.intersections.plot_intersections_2dyz()
    if settings.recalculate_gram_matrix:
        print(setup.number_of_linegroups, 'groups of parallel lines found')

        print('making GramMatrix')
        setup.MakeGramMatrix()
        #np.save('..\Output\calculations_'+settings.Name_of_calculation +'\setup.npy', setup)
    return(setup)


if __name__ == "__main__":
    setup = Measurement_setup("../Setups/3D_setup5.csv")
    print(setup.line_count, 'lines constructed')
    #setup.plot_setup()
    print('Grouping lines together...')
    setup.MakeLineGroups()
    print(setup.number_of_linegroups, 'groups of parallel lines found')
    print('calculating intersections')
    setup.SetIntersections()
    setup.intersections.plotIntersections()
    print('making tubes')
    setup.MakeTubes(width= 0.03)
    print('making GramMatrix')
    setup.MakeGramMatrix()
