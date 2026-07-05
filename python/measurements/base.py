"""
Base Measurement Class
"""

from abc import ABC, abstractmethod


class Measurement(ABC):

    @abstractmethod
    def compute(self, results):
        pass