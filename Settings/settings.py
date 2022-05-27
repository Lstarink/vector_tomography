# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:42:47 2022

@author: lstar
"""

"""All settings go in this file"""
import sympy as sp

"""settings for what part of code needs to be rerun"""
only_calculate_setup = False
recalculate_gram_matrix = False
generate_your_own_measurement = True

"""Define what setup you want to use
 Define what measurements you want to use
 Define where you want to save intersections and Gram Matrix if recalculating the setup"""
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')

u = (sp.Float(0.1)) * (y)
v = (sp.Float(0.1))* (x)
w = (sp.Float(0.01))
 
FileName = 'final_setup.csv'
Name_of_calculation = 'Numerical_0.02@50'
generated_measurement_file = 'Speeds' + Name_of_calculation + '.npy'
if generate_your_own_measurement:
    measurement_file = generated_measurement_file
else:
    measurement_file = 'alleen_recht_ref.npy'

"""Settings for intersections"""
intersection_boundary_edge_x = 0.005 #meters, defines a boundary layer around the edge of the setup for which intersections will not be included, generally to exclude the sensors as intersections
intersection_boundary_edge_y = 0.005
intersection_boundary_edge_z = 0.000
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
interpolation_offset_x = 0.01 #m
interpolation_offset_y = 0.01 #m
interpolation_offset_z = 0.01 #m

plot_interpolated_resolution = 11
plot_amount_of_interpolated_slices = 3

"""Settings for plotting"""
save_figures = True
plot_line_setup = False
plot_tube_intersections = False and use_integration_for_gram_matrix
plot_line_intersections = False
plot_tube_setup = False
plot_tube_setup_2d = False
plot_u0 = False
plot_original_field = False
plot_intersection_field = False
plot_interpolated = False
plot_error = False and generate_your_own_measurement
plot_field_sliced = False
plot_error_sliced = False and generate_your_own_measurement

calculate_error = True and generate_your_own_measurement


