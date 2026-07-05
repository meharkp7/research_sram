"""
Reusable CMOS Inverter
Framework V2
"""

from core.circuit import Circuit
from core.net import Net
from core.transistor import NMOS, PMOS
from core.voltage import VoltageSource
from core.parameters import ParameterSet


class CMOSInverter(Circuit):

    def __init__(
        self,
        input_node=None,
        output_node=None,
        vdd_node=None,
        gnd_node=None,
        parameters=None,
        own_supply=True
    ):

        super().__init__("CMOS Inverter")

        if parameters is None:
            parameters = ParameterSet()

        self.parameters = parameters

        # -----------------------------------
        # Nets
        # -----------------------------------

        self.vdd = vdd_node if vdd_node else Net("VDD")
        self.gnd = gnd_node if gnd_node else Net("0")

        self.vin = input_node if input_node else Net("VIN")
        self.vout = output_node if output_node else Net("VOUT")

        # -----------------------------------
        # Power Supply
        # -----------------------------------

        if own_supply:

            self.add(

                VoltageSource(
                    "VDD",
                    self.vdd,
                    self.gnd,
                    self.parameters.vdd
                )

            )

        # -----------------------------------
        # PMOS
        # -----------------------------------

        self.pmos = PMOS(

            "M1",

            drain=self.vout,

            gate=self.vin,

            source=self.vdd,

            bulk=self.vdd,

            width=self.parameters.wp,

            length=self.parameters.length

        )

        # -----------------------------------
        # NMOS
        # -----------------------------------

        self.nmos = NMOS(

            "M2",

            drain=self.vout,

            gate=self.vin,

            source=self.gnd,

            bulk=self.gnd,

            width=self.parameters.wn,

            length=self.parameters.length

        )

        self.add(self.pmos)

        self.add(self.nmos)