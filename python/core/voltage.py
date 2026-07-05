"""
Independent DC Voltage Source
"""

from core.net import Net


class VoltageSource:

    def __init__(self,
                 name: str,
                 positive: Net,
                 negative: Net,
                 voltage: float):

        self.name = name
        self.positive = positive
        self.negative = negative
        self.voltage = voltage

    def spice(self):

        return (
            f"{self.name} "
            f"{self.positive} "
            f"{self.negative} "
            f"DC {self.voltage}"
        )