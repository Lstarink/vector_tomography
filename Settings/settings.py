# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:42:47 2022

@author: lstar
"""

"""All settings go in this file"""

"""settings for what part of code needs to be rerun"""
only_calculate_setup = False
recalculate_gram_matrix = True
generate_your_own_measurement = True

"""Define what setup you want to use
 Define what measurements you want to use
 Define where you want to save intersections and Gram Matrix if recalculating the setup"""
 
FileName = '3D_setup2.csv'
Name_of_calculation = 'setup1'
generated_measurement_file = 'Speeds' + Name_of_calculation + '.npy'
measurement_file = generated_measurement_file

"""Settings for intersections"""
intersection_boundary_edge = 0.005 #meters, defines a boundary layer around the edge of the setup for which intersections will not be included, generally to exclude the sensors as intersections
use_only_full_rank_intersections = False

"""Settings for error of the sensors"""
use_sensor_error = False
sensor_stddev = 0.00166 #meters

"""Settings for Gram Matrix"""
use_integration_for_gram_matrix = False
matrix_integration_setting = 75 #If used needs alot of calculation time, and value has to be set to at least 100
tube_width = 0.02

"""Settings for reconstruction of the field"""
line_integral_iteration_steps = 100

"""Settings for interpolation"""
plot_interpolated_resolution = 11
plot_amount_of_interpolated_slices = 3

"""Settings for plotting"""
plot_line_setup = False
plot_tube_intersections = False and use_integration_for_gram_matrix
plot_line_intersections = False
plot_tube_setup = False
plot_tube_setup_2d = False
plot_u0 = False
plot_original_field = False
plot_intersection_field = False
plot_interpolated = False
plot_error = False
plot_error_sliced = False


