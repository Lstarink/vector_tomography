# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:42:47 2022

@author: lstar
"""

"""All settings go in this file"""

"""settings for what part of code needs to be rerun"""
only_calculate_setup = False
recalculate_gram_matrix = True
generate_your_own_measurement = False #It is possible to define your own field and compare it to a field reconstructed from real measurements. Set measurement_file to your measurements and this to True

"""Define what setup you want to use
 Define what measurements you want to use
 Define where you want to save intersections and Gram Matrix if recalculating the setup"""
 
FileName = 'ZONDER_OBSTAKEL_Alleen_recht.csv'
Name_of_calculation = 'alleen_recht'
generated_measurement_file = 'Speeds' + Name_of_calculation + '.npy'
if generate_your_own_measurement:
    measurement_file = generated_measurement_file
else:
    measurement_file = 'alleen_recht_ref.npy'

"""Settings for intersections"""
intersection_boundary_edge = 0.005 #meters, defines a boundary layer around the edge of the setup for which intersections will not be included, generally to exclude the sensors as intersections
grid_size_offset=None
use_only_full_rank_intersections = False

"""Settings for error of the sensors"""
use_sensor_error = False
sensor_stddev = 0.00166 #meters

"""Settings for scaling"""
if generate_your_own_measurement:
    quiver_scale = None
else:
    quiver_scale = 200

"""Settings for Gram Matrix"""
use_integration_for_gram_matrix = True
matrix_integration_setting = 20 #If used needs alot of calculation time, and value has to be set to at least 100
tube_width = 0.02 #m

"""Settings for reconstruction of the field"""
line_integral_iteration_steps = 100

"""Settings for interpolation"""
interpolation_offset = 0.03 #m
plot_interpolated_resolution = 11
plot_amount_of_interpolated_slices = 5

"""Settings for plotting"""
plot_line_setup = False
plot_tube_intersections = False and use_integration_for_gram_matrix
plot_line_intersections = True
plot_tube_setup = False
plot_tube_setup_2d = False
plot_u0 = False
plot_original_field = False
plot_intersection_field = False
plot_interpolated = False
plot_error = False and generate_your_own_measurement
plot_field_sliced = True
plot_error_sliced = True and generate_your_own_measurement


