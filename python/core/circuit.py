"""
Core Circuit Class
Supports:
- Components
- Child Circuits (Hierarchical Design)
- SPICE Directives
"""


class Circuit:

    def __init__(self, name: str):

        self.name = name

        # Primitive devices
        self.components = []

        # Child circuits
        self.subcircuits = []

        # SPICE directives
        self.directives = []

    # -------------------------
    # Components
    # -------------------------

    def add(self, component):

        self.components.append(component)

    # -------------------------
    # Child Circuits
    # -------------------------

    def add_subcircuit(self, circuit):

        self.subcircuits.append(circuit)

    # -------------------------
    # SPICE Directives
    # -------------------------

    def directive(self, text):

        self.directives.append(text)

    # -------------------------
    # Collect Components
    # -------------------------

    def all_components(self):

        devices = []

        devices.extend(self.components)

        for child in self.subcircuits:
            devices.extend(child.all_components())

        return devices

    # -------------------------
    # Collect Directives
    # -------------------------

    def all_directives(self):

        directives = []

        directives.extend(self.directives)

        for child in self.subcircuits:
            directives.extend(child.all_directives())

        return directives