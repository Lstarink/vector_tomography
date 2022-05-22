# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:42:47 2022

@author: lstar
"""

"""All settings go in this file"""

"""settings for what part of code needs to be rerun"""
#recalculate_setup =  False
recalculate_gram_matrix = False
generate_your_own_measurement = True

"""Define what setup you want to use
 Define what measurements you want to use
 Define where you want to save intersections and Gram Matrix if recalculating the setup"""
 
FileName = 'final_setup.csv'
Name_of_calculation = 'calculations_final_numerical'
generated_measurement_file = 'Speeds' + Name_of_calculation + '.npy'
measurement_file = generated_measurement_file

"""Settings for intersections"""
intersection_boundary_edge = 0.005 #meters, defines a boundary layer around the edge of the setup for which intersections will not be included, generally to exclude the sensors as intersections
use_only_full_rank_intersections = False

"""Settings for error of the sensors"""
use_sensor_error = True
sensor_stddev = 0.000166 #meters

"""Settings for Gram Matrix"""
use_integration_for_gram_matrix = False
matrix_integration_setting = 25 #If used needs alot of calculation time, and value has to be set to at least 100
tube_width = 0.02

"""Settings for reconstruction of the field"""
line_integral_iteration_steps = 100

"""Settings for interpolation"""
plot_interpolated_resolution = 11
plot_amount_of_interpolated_slices = 3
slice_height_z = 0.5

"""Settings for plotting"""
plot_line_setup = False
plot_tube_intersections = False and use_integration_for_gram_matrix
plot_line_intersections = False and use_integration_for_gram_matrix
plot_tube_setup = False
plot_u0 = False
plot_original_field = False
plot_intersection_field = False
plot_interpolated = True
plot_error = True
plot_error_sliced = True


