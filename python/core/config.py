"""
Technology Configuration
"""

class Technology:

    def __init__(self):

        self.vdd = 1.8

        self.nmos_model = (
            ".model MyNMOS NMOS "
            "(LEVEL=1 VTO=0.6 KP=250u LAMBDA=0.02)"
        )

        self.pmos_model = (
            ".model MyPMOS PMOS "
            "(LEVEL=1 VTO=-0.6 KP=120u LAMBDA=0.02)"
        )

TECH = Technology()