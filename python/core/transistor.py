class MOSFET:

    def __init__(
        self,
        name,
        drain,
        gate,
        source,
        bulk,
        model,
        width=1e-6,
        length=180e-9
    ):

        self.name = name
        self.drain = drain
        self.gate = gate
        self.source = source
        self.bulk = bulk
        self.model = model

        self.width = width
        self.length = length

    def spice(self):

        return (
            f"{self.name} "
            f"{self.drain} "
            f"{self.gate} "
            f"{self.source} "
            f"{self.bulk} "
            f"{self.model} "
            f"L={self.length} "
            f"W={self.width}"
        )
    
class PMOS(MOSFET):

    def __init__(
        self,
        name,
        drain,
        gate,
        source,
        bulk,
        width=2e-6,
        length=180e-9
    ):

        super().__init__(
            name,
            drain,
            gate,
            source,
            bulk,
            "MyPMOS",
            width,
            length
        )

class NMOS(MOSFET):

    def __init__(
        self,
        name,
        drain,
        gate,
        source,
        bulk,
        width=1e-6,
        length=180e-9
    ):

        super().__init__(
            name,
            drain,
            gate,
            source,
            bulk,
            "MyNMOS",
            width,
            length
        )