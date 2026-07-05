"""
Base Experiment Class
"""

from abc import ABC, abstractmethod


class Experiment(ABC):

    def __init__(self, circuit):

        self.circuit = circuit

        self.results = None

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def analyze(self):
        pass