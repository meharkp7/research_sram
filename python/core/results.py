"""
Simulation Results
"""


class Results:

    def __init__(self):

        self.voltages = {}

        self.currents = {}

        self.measurements = {}

    def add_voltage(self, name, values):

        self.voltages[name] = values

    def add_current(self, name, values):

        self.currents[name] = values

    def add_measurement(self, name, value):

        self.measurements[name] = value

    def summary(self):

        print("="*50)

        print("RESULT SUMMARY")

        print("="*50)

        print()

        print("Voltages")

        print(self.voltages.keys())

        print()

        print("Currents")

        print(self.currents.keys())

        print()

        print("Measurements")

        for k, v in self.measurements.items():

            print(f"{k} : {v}")