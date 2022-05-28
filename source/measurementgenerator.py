

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import csv
from mpl_toolkits.mplot3d import Axes3D
import settings
import vector_field


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
    
    "numericaly integrates a vector field over a line from A to B with iteration_steps steps"
    def V_average(self, field):
        iteration_steps = 50
        ds = self.length/iteration_steps
        
        v_sum = 0
        
        for i in range(0, iteration_steps):
            location_x = self.A[0] + self.unit_vector[0]*ds*i
            location_y = self.A[1] + self.unit_vector[1]*ds*i
            location_z = self.A[2] + self.unit_vector[2]*ds*i
            
            v = field.Sample(location_x, location_y, location_z) #U(x,y,z)
            
            v_projected = np.dot(v,self.unit_vector)
            v_sum += v_projected
            
        v_average = v_sum/iteration_steps
    
        return(v_average)
    
class Measurement_setup:
    def __init__(self, file):
        self.file_name = '../Setups/' + file
        self.lines = Measurement_setup.MakeLines(self)
        self.line_count = 0
    
    "Loops through the rows of the input file, constructs a line for every row of the file, and appends the line to lines"
    def MakeLines(self):
        lines = []
        line_count = 0
        
        with open(self.file_name) as setup:
            csv_reader = csv.reader(setup, delimiter=',')
            for row in csv_reader:
                lines.append(line([float(row[1]),float(row[2]),float(row[3])],
                                  [float(row[4]),float(row[5]),float(row[6])]))
    
                line_count += 1
            
            self.line_count = line_count
        return(lines)
    
    def CalculateRange(self):
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        z_min = 0
        z_max = 0
        
        
        for i in range(len(self.lines)):
            #set minima x
            if (self.lines[i].A[0] < x_min):
                x_min = self.lines[i].A[0]
            if (self.lines[i].B[0] < x_min):
                x_min = self.lines[i].B[0]
            #set minima y
            if (self.lines[i].A[1] < y_min):
                y_min = self.lines[i].A[1]
            if (self.lines[i].B[1] < y_min):
                y_min = self.lines[i].B[1]
            #set minima z
            if (self.lines[i].A[2] < z_min):
                z_min = self.lines[i].A[2]
            if (self.lines[i].B[2] < z_min):
                z_min = self.lines[i].B[2]
                
            #set maxima x
            if (self.lines[i].A[0] > x_max):
                x_max = self.lines[i].A[0]
            if (self.lines[i].B[0] > x_max):
                x_max = self.lines[i].B[0]
            #set maxima y
            if (self.lines[i].A[1] > y_max):
                y_max = self.lines[i].A[1]
            if (self.lines[i].B[1] > y_max):
                y_max = self.lines[i].B[1]
            #set maxima z
            if (self.lines[i].A[2] > z_max):
                z_max = self.lines[i].A[2]
            if (self.lines[i].B[1] > z_max):
                z_max = self.lines[i].B[2]
    
        return(x_min, y_min, z_min, x_max, y_max, z_max)
    
    def SaveMeasurements(self,field, outputfile):
        row_count = 0
        with open(self.file_name, 'r') as read_obj, \
             open(outputfile, 'w', newline='') as write_obj:
                 csv_reader = csv.reader(read_obj)
                 csv_writer = csv.writer(write_obj)
                 for row in csv_reader:
                    row.append(self.lines[row_count-1].V_average(field))
                    row.append(self.lines[row_count-1].unit_vector[0])
                    row.append(self.lines[row_count-1].unit_vector[1])
                    row.append(self.lines[row_count-1].unit_vector[2])
                    row.append(self.lines[row_count-1].unit_vector)
                    csv_writer.writerow(row)
                    row_count += 1
        return()
    
    def Save_speeds(self, field):
        
        V_average = np.zeros(len(self.lines))
        error = np.zeros([len(self.lines)])
        
        if (settings.use_sensor_error == True):
            for i in range(len(self.lines)):
                error[i] = np.random.normal(0, settings.sensor_stddev)
                            
        
        for i in range(len(self.lines)):
            V_average[i] = self.lines[i].V_average(field) + error[i]
        
        np.save('../Measurements/'+ settings.generated_measurement_file, V_average)
    
    def plot_setup(self):
        fig = plt.figure(figsize=(4,4))

        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(self.lines)):
            ax.plot([self.lines[i].A[0],self.lines[i].B[0]],
                     [self.lines[i].A[1],self.lines[i].B[1]], 
                     [self.lines[i].A[2],self.lines[i].B[2]], color = 'b')  
        plt.show()
        return()
    

def instantiate_field1():
    u = settings.u
    v = settings.v
    w = settings.w
    return(u,v,w)

def plotIntersectionField(field, intersections, gridSize):
    x = intersections[0]
    y = intersections[1]
    z = intersections[2]
    
   
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, field[:,0], field[:,1], field[:,2])
    
    ax.set_xlim([gridSize[0], gridSize[3]])
    ax.set_ylim([gridSize[1], gridSize[4]])
    ax.set_zlim([gridSize[2], gridSize[5]])

    plt.show()  

def Generate_V_average(setup):
    #settings
    intersections = np.load('../Output/' + 'calculations_'+ settings.Name_of_calculation + '/Intersections.npy')

    
    #Make a vector field
    field1 = vector_field.vector_field(settings.u, settings.v, settings.w)
    
    #Load setup
    measurement_setup = Measurement_setup(settings.FileName)
    x_min, y_min, z_min, x_max, y_max, z_max = measurement_setup.CalculateRange()
    gridSize = np.array([x_min, y_min,z_min, x_max, y_max, z_max])
    measurement_setup.Save_speeds(field1)
    field_sampled_intersections = np.zeros([len(intersections[0]), 3])
    for i in range(len(intersections[0])):
        field_sampled_intersections[i][:] = field1.Sample(intersections[0][i], intersections[1][i], intersections[2][i])

    if settings.plot_original_field:
        plotIntersectionField(field_sampled_intersections, intersections, gridSize)
    return(field_sampled_intersections, field1)
       
if __name__ == "__main__":
    Generate_V_average()