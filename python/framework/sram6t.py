"""
6T SRAM Cell
Framework V2
"""

from core.circuit import Circuit
from core.net import Net
from core.transistor import NMOS
from core.voltage import VoltageSource

from framework.cross_coupled import CrossCoupledPair


class SRAM6T(Circuit):

    def __init__(self, parameters=None):

        super().__init__("6T SRAM")

        self.vdd = Net("VDD")
        self.gnd = Net("0")

        self.q = Net("Q")
        self.qb = Net("QB")

        self.bl = Net("BL")
        self.blb = Net("BLB")

        self.wl = Net("WL")

        self.add(

            VoltageSource(

                "VDD",

                self.vdd,

                self.gnd,

                1.8

            )

        )

        self.cell = CrossCoupledPair(

            self.q,

            self.qb,

            self.vdd,

            self.gnd,

            parameters

        )

        self.add_subcircuit(self.cell)

        self.left_access = NMOS(

            "AX1",

            drain=self.bl,

            gate=self.wl,

            source=self.q,

            bulk=self.gnd

        )

        self.right_access = NMOS(

            "AX2",

            drain=self.blb,

            gate=self.wl,

            source=self.qb,

            bulk=self.gnd

        )

        self.add(self.left_access)

        self.add(self.right_access)