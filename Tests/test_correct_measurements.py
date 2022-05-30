import unittest
import correct_measurements

class TestCorrectMeasurements(unittest.TestCase):
    def test_measurement(self):
        heen = 0.200
        terug = 0.201

        no_obstacle_forth_array = 0
        no_obsacle_back_array = 0
        obstacle_forth_array = 0
        obstacle_back_array = 0

        self.assertAlmostEqual(0,correct_measurements.CorrectMeasurements(no_obstacle_forth_array, no_obsacle_back_array, obstacle_forth_array, obstacle_back_array))