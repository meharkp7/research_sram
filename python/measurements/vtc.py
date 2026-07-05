from measurements.gain import Gain
from measurements.switching_voltage import SwitchingVoltage


class VTCMeasurement:

    def compute(self, results):

        vin = results.voltages["vin"]

        vout = results.voltages["vout"]

        vm = SwitchingVoltage.compute(
            vin,
            vout
        )

        gain = Gain.compute(
            vin,
            vout
        )

        return {

            "switching_voltage": vm,

            "maximum_gain": gain["gain"],

            "gain_point": gain["vm"]

        }