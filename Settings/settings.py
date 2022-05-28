# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:42:47 2022

@author: lstar
"""

"""All settings go in this file"""
import sympy as sp

"""Settings for what needs to be run"""
only_calculate_setup = False
recalculate_gram_matrix = False
generate_your_own_measurement = True

"""
Settings for what is used en where to save it
"""
FileName = 'final_setup.csv'
Name_of_calculation = 'final_setup_0.02@20_divergence_field'
generated_measurement_file = 'Speeds' + Name_of_calculation + '.npy'
if generate_your_own_measurement:
    measurement_file = generated_measurement_file
else:
    measurement_file = 'ZONDER_OBSTAKEL_back_and_forward.npy'

"""Define your own vector field"""
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')

u = -(sp.Float(0.1)) * (y-sp.Float(0.101))
v = -(sp.Float(0.1))* (x-sp.Float(0.101))
w = (sp.Float(0.3))*(z)

"""Settings for intersections"""
intersection_boundary_edge_x = 0.005 #meters, defines a boundary layer around the edge of the setup for which intersections will not be included, generally to exclude the sensors as intersections
intersection_boundary_edge_y = 0.005
intersection_boundary_edge_z = 0.000
use_only_full_rank_intersections = False

"""Settings for error of the sensors"""
use_sensor_error = False
sensor_stddev = 0.00166 #meters


"""Settings for Gram Matrix"""
use_integration_for_gram_matrix = True
matrix_integration_setting = 20 #If used needs alot of calculation time, and value has to be set to at least 100
tube_width = 0.02 #m

"""Settings for interpolation"""
interpolation_offset_x = 0.03 #m
interpolation_offset_y = 0.03 #m
interpolation_offset_z = 0.00 #m


"""Settings for plotting"""
#Pas alleen True of False aan!
plot_line_setup = False
plot_line_intersections = False
plot_tube_setup = False
plot_tube_setup_2d = False
plot_u0 = False
plot_original_field = False
plot_intersection_field = False
plot_interpolated = False
plot_error = False and generate_your_own_measurement

save_figures = True #Letop hij slaat alleen de plotjes op als je de plot instelling ook op True hebt staan.
plot_interpolated_resolution = 3   #bepaalt hoeveel pijlen er worden geplot. 11 houdt het overzichtelijk vindt ik, Maar hier kun je zelf mee spelen.
plot_amount_of_interpolated_slices = 3 #Bepaalt hoeveel slices je te zien krijgt in x y en z richting
plot_field_sliced = True
plot_error_sliced = True and generate_your_own_measurement
show_sliced = False and plot_error_sliced
calculate_error = True and generate_your_own_measurement

###Settings hieronder hebben jullie niet nodig!!

"""Settings for reconstruction of the field"""
line_integral_iteration_steps = 100

"""Settings for scaling"""
if generate_your_own_measurement:
    quiver_scale = None
else:
    quiver_scale = 133

plot_tube_intersections = False and use_integration_for_gram_matrix