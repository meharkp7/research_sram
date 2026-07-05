import numpy as np


class Gain:

    @staticmethod
    def compute(vin, vout):

        slope = np.gradient(vout, vin)

        gain = np.min(slope)

        index = np.argmin(slope)

        vm = vin[index]

        return {
            "gain": gain,
            "vm": vm
        }