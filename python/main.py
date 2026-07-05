from framework.sram6t import SRAM6T
from core.spice_writer import SpiceWriter

cell = SRAM6T()

writer = SpiceWriter(cell)

writer.write("../spice/sram6t.sp")

print("SRAM Netlist Generated")