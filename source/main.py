# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:53:08 2022

@author: lstar
"""
import numpy as np
import make_setup
import measurementgenerator
import measurement_specific
import settings
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    """load_data.Make_setup makes an instance of Measurement_Setup. Depending on if recalculate_setup in the settings is on it will 
    calculate all attributes of a measurement setup that are independent of measurements, primarily the Gram Matrix and the line intersection points
    If recalculate_setup is False it will not calculate a new Gram Matrix and intersections and saved versions of these will be used in next functions
    to save calculation time."""
    setup = make_setup.Make_Setup('../Setups/' + settings.FileName)

    if not settings.only_calculate_setup:
        """measurementgenerator.Generate_V_average makes"""
        if settings.generate_your_own_measurement:
            sampled_field, vector_field = measurementgenerator.Generate_V_average(setup)
            final_field = measurement_specific.make_measurement_calculation(setup, sampled_field, vector_field)
        else:
            final_field = measurement_specific.make_measurement_calculation(setup)






