"""
Global Simulation Parameters
"""


class ParameterSet:

    def __init__(self):

        # Supply

        self.vdd = 1.8

        # Input

        self.vin = 0

        # Device Geometry

        self.wp = 2e-6

        self.wn = 1e-6

        self.length = 180e-9

        # Environment

        self.temperature = 27

        self.process = "TT"

        # Adaptive Bias

        self.body_bias = 0

        # Monte Carlo

        self.seed = None

    def __str__(self):

        return vars(self).__repr__()