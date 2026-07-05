"""
DC Transfer Experiment
"""

from experiments.base_experiment import BaseExperiment

from core.spice_writer import SpiceWriter
from core.simulator import Simulator
from measurements.vtc import VTCMeasurement
from measurements.switching_voltage import SwitchingVoltage
from research_sram.python.core.voltage import VoltageSource
from core.voltage import VoltageSource

class DCTransferExperiment(BaseExperiment):

    def __init__(
        self,
        circuit,
        filename="../spice/03_inverter/dc_transfer.spice"
    ):

        super().__init__(circuit)

        self.filename = filename

    def prepare(self):

        self.circuit.add(
            VoltageSource(
                "V2",
                self.circuit.vin,
                self.circuit.gnd,
                0
            )
        )

        self.circuit.directive(".dc V2 0 1.8 0.01")

        writer = SpiceWriter(self.circuit)

        writer.write(self.filename)

    def run(self):

        sim = Simulator(self.filename)

        self.results = sim.run_ngspice()

        return self.results

    def analyze(self):

        print("Analyzing...")

        vtc = VTCMeasurement()

        vin = self.results.voltages["vin"]

        vout = self.results.voltages["vout"]

        vm = SwitchingVoltage.compute(vin, vout)

        metrics = {
            "switching_voltage": vm
        }

        self.results.add_measurement(
            "VTC",
            metrics
        )