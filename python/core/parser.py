"""
Raw Data Parser
"""

from pathlib import Path


class Parser:

    def __init__(self, raw_file):

        self.raw_file = Path(raw_file)

    def exists(self):

        return self.raw_file.exists()

    def info(self):

        if self.exists():

            print("Raw file found!")

            print(self.raw_file)

        else:

            print("Raw file not found.")