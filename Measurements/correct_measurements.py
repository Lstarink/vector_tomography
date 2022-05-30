import numpy as np

def CorrectMeasurements(no_obstacle_forth_array, no_obsacle_back_array):
    c0 = 343
    c1 = 343

    assert(no_obstacle_forth_array.size == no_obsacle_back_array.size == obstacle_forth_array.size == obstacle_back_array.size)

    for forth, back in zip(no_obstacle_forth_array, no_obsacle_back_array):
        tof1 = forth/c0
        tof2 = back/c0

        c_corrected = (tof1/tof2)*c1

    return(c_corrected)


c_corrected = CorrectMeasurements(np.load(ZONDER_OBSTAKEL_ref.npy))
