"""
Cross Coupled CMOS Inverters
Framework V2
"""

from core.circuit import Circuit

from framework.inverter import CMOSInverter


class CrossCoupledPair(Circuit):

    def __init__(
        self,
        q,
        qb,
        vdd,
        gnd,
        parameters=None
    ):

        super().__init__("Cross Coupled Pair")

        self.left = CMOSInverter(

            input_node=qb,

            output_node=q,

            vdd_node=vdd,

            gnd_node=gnd,

            parameters=parameters,

            own_supply=False

        )

        self.right = CMOSInverter(

            input_node=q,

            output_node=qb,

            vdd_node=vdd,

            gnd_node=gnd,

            parameters=parameters,

            own_supply=False

        )

        self.add_subcircuit(self.left)

        self.add_subcircuit(self.right)