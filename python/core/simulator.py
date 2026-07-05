"""
SPICE Simulation Backend
"""

import subprocess
from pathlib import Path
from core.results import Results
from core.raw_reader import RawReader

class Simulator:

    def __init__(self, spice_file):

        self.spice_file = Path(spice_file)

    def run_ngspice(self):

        print("Running ngspice...")

        raw_file = self.spice_file.with_suffix(".raw")

        result = subprocess.run(
            [
                "ngspice",
                "-b",
                "-r",
                str(raw_file),
                str(self.spice_file)
            ],
            capture_output=True,
            text=True
        )

        print(result.stdout)

        if result.returncode != 0:

            print(result.stderr)

            raise RuntimeError("Simulation Failed")

        print("Simulation Finished")

        results = Results()

        raw_file = self.spice_file.with_suffix(".raw")

        reader = RawReader(raw_file)

        results.add_voltage(
            "vin",
            reader.wave("v(vin)")
        )

        results.add_voltage(
            "vout",
            reader.wave("v(vout)")
        )

        results.add_current(
            "i_vdd",
            reader.wave("i(v1)")
        )

        return results