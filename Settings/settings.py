# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:42:47 2022

@author: lstar
"""

"""All settings go in this file"""

"""settings for what part of code needs to be rerun"""
#recalculate_setup =  False
recalculate_gram_matrix_and_intersections = True
generate_your_own_measurement = True

"""Define what setup you want to use
 Define what measurements you want to use
 Define where you want to save intersections and Gram Matrix if recalculating the setup"""
 
FileName = '3D_setup4.csv'
Name_of_calculation = 'setup4_numerical'
generated_measurement_file = 'Speeds' + Name_of_calculation + '.npy'
measurement_file = generated_measurement_file

"""Settings for error of the sensors"""
use_sensor_error = True
sensor_stddev = 0.000166 #meters

"""Settings for Gram Matrix"""
use_integration_for_gram_matrix = False
matrix_integration_setting = 20 #If used needs alot of calculation time, and value has to be set to at least 100
tube_width = 0.0002

"""Settings for reconstruction of the field"""
line_integral_iteration_steps = 100

"""Settings for interpolation"""
plot_interpolated_resolution = 11
plot_interpolated_slices = 4
slice_height_z = 0.5

"""Settings for plotting"""
plot_line_setup = False
plot_tube_intersections = False and use_integration_for_gram_matrix
plot_line_intersections = True and recalculate_gram_matrix_and_intersections
plot_tube_setup = False
plot_u0 = False
plot_interpolated = False
plot_error = False

