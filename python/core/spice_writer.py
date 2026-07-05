"""
SPICE Netlist Writer
Framework V2
"""

from pathlib import Path


class SpiceWriter:

    def __init__(self, circuit):

        self.circuit = circuit

    def write(self, filename):

        filename = Path(filename)

        lines = []

        # ---------------------------------
        # Header
        # ---------------------------------

        lines.append(f"* {self.circuit.name}")
        lines.append("")

        # ---------------------------------
        # Device Models
        # ---------------------------------

        lines.append(
            ".model MyNMOS NMOS (LEVEL=1 VTO=0.6 KP=250u LAMBDA=0.02)"
        )

        lines.append(
            ".model MyPMOS PMOS (LEVEL=1 VTO=-0.6 KP=120u LAMBDA=0.02)"
        )

        lines.append("")

        # ---------------------------------
        # Components
        # ---------------------------------

        for component in self.circuit.all_components():

            lines.append(component.spice())

        lines.append("")

        # ---------------------------------
        # Directives
        # ---------------------------------

        for directive in self.circuit.all_directives():

            lines.append(directive)

        lines.append("")

        # ---------------------------------
        # End
        # ---------------------------------

        lines.append(".end")

        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w") as f:

            f.write("\n".join(lines))