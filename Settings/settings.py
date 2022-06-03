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
Name_of_calculation = 'final_setup_0.0225@100'
generated_measurement_file = 'Speeds' + Name_of_calculation + '.npy'
if generate_your_own_measurement:
    measurement_file = generated_measurement_file
else:
    measurement_file = 'ZONDER_OBSTAKEL_back_and_forward.npy'

"""Define your own vector field"""
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')

n1 = 3
n2 = 3
theta = sp.pi/4

x_ = sp.cos(theta)*(x-0.101) - sp.sin(theta)*(y-0.101)
y_ = sp.sin(theta)*(x-0.101) + sp.cos(theta)*(y-0.101)

u_ = ((sp.Float(0.3)) * y_ + sp.Float(0.6)*sp.cos(sp.pi*n2*y_/0.101))
v_ = (-(sp.Float(0.3))*x_ +sp.Float(0.6)*sp.sin(sp.pi*n2*x_/0.101))
w = (sp.Float(4.0))

u = sp.cos(theta+sp.pi/2)*u_ - sp.sin(theta+sp.pi/2)*v_
v = sp.sin(theta+sp.pi/2)*u_ + sp.cos(theta+sp.pi/2)*v_


"""Settings for intersections"""
intersection_boundary_edge_x = 0.005 #meters, defines a boundary layer around the edge of the setup for which intersections will not be included, generally to exclude the sensors as intersections
intersection_boundary_edge_y = 0.005
intersection_boundary_edge_z = 0.000
use_only_full_rank_intersections = False

"""Settings for error of the sensors"""
use_sensor_error = False
sensor_stddev = 0.000455 #meters
temperature_increase = 0.5 #kelvin


"""Settings for Gram Matrix"""
use_integration_for_gram_matrix = True
matrix_integration_setting = 20 #If used needs alot of calculation time, and value has to be set to at least 100
tube_width = 0.0225 #m

"""Settings for interpolation"""
interpolation_offset_x = 0.03 #m
interpolation_offset_y = 0.03 #m
interpolation_offset_z = 0.0 #m


"""Settings for plotting"""
#Pas alleen True of False aan!
plot_line_setup = False
plot_line_intersections = False
plot_tube_setup = False
plot_tube_setup_2d = False
plot_u0 = False
plot_original_field = False
plot_intersection_field = False
plot_interpolated = True
plot_error = False and generate_your_own_measurement

save_figures = True #Letop hij slaat alleen onderstaande plotjes op en slaat ze alleen op als je de plot instelling ook op True hebt staan.
only_combined = False
plot_interpolated_resolution = 25  #bepaalt hoeveel pijlen er worden geplot. 11 houdt het overzichtelijk vindt ik, Maar hier kun je zelf mee spelen.
plot_amount_of_interpolated_slices = 3 #Bepaalt hoeveel slices je te zien krijgt in x y en z richting
inplane_error = True
arrow_legenda = 5 #Bepaalt hoe groot het legenda pijltje rechtsbovenin is bij de plotjes. Zorg dat het dezelfde orde van grote heeft als je vector veld!
arrow_legenda_string = r'$5.0\frac{m}{s}$' # Vul hier in wat je bij de regel hierboven heb gezet

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
    quiver_scale = 150

plot_tube_intersections = False and use_integration_for_gram_matrix