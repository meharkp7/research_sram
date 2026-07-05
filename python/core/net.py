"""
Represents an electrical node (wire) in a circuit.
"""


class Net:

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name