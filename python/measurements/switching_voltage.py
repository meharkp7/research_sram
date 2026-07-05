import numpy as np


class SwitchingVoltage:

    @staticmethod
    def compute(vin, vout):

        error = np.abs(vin - vout)

        index = np.argmin(error)

        return vin[index]