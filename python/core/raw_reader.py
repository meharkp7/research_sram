"""
Universal RAW Reader
Works with ngspice/LTspice through spicelib
"""

from spicelib import RawRead


class RawReader:

    def __init__(self, filename):

        self.raw = RawRead(filename)

    def traces(self):

        try:
            return self.raw.get_trace_names()

        except Exception:
            return self.raw.get_trace_list()

    def wave(self, name):

        trace = self.raw.get_trace(name)

        try:
            return trace.get_wave()

        except Exception:
            return trace.get_wave(0)

    def axis(self):

        if "time" in self.traces():
            return self.wave("time")

        try:
            return self.raw.get_axis().get_wave()

        except Exception:
            return self.raw.get_axis()