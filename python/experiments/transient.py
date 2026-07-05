"""
Transient Experiment
"""

from experiments.base_experiment import Experiment

from core.spice_writer import SpiceWriter
from core.simulator import Simulator
from core.raw_reader import RawReader

import matplotlib.pyplot as plt


class TransientExperiment(Experiment):

    def __init__(
        self,
        circuit,
        filename="../spice/03_inverter/transient.sp"
    ):

        super().__init__(circuit)

        self.filename = filename

    def prepare(self):

        # Input Pulse
        self.circuit.directive(
            "VIN VIN 0 PULSE(0 1.8 0 10p 10p 2n 4n)"
        )

        # Transient Analysis
        self.circuit.directive(
            ".tran 10p 20n"
        )

        writer = SpiceWriter(self.circuit)
        writer.write(self.filename)

    def run(self):

        sim = Simulator(self.filename)

        self.results = sim.run_ngspice()

        return self.results

    def analyze(self):

        print()
        print("Transient Analysis Complete")

        reader = RawReader("../spice/03_inverter/transient.raw")

        print("\nAvailable Traces")
        print("----------------")

        for trace in reader.traces():
            print(trace)

        # Read waveforms
        time = reader.wave("time")
        vin = reader.wave("v(vin)")
        vout = reader.wave("v(vout)")

        print("\nPoints :", len(time))

        # Plot
        plt.figure(figsize=(10, 5))

        plt.plot(time, vin, label="VIN", linewidth=2)
        plt.plot(time, vout, label="VOUT", linewidth=2)

        plt.title("CMOS Inverter Transient Response")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()