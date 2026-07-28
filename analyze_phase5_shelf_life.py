import subprocess
import numpy as np

VTH0_REF_NOMINAL = 0.46893
W_UM = 3.2
THRESHOLD = 0.6614  # from Segment 5.6's calibration run

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w={w_nm:.0f}n
Mtarget d2 0 0 0 nmos_target l=45n w={w_nm:.0f}n

Vref    d1 0 1.0
Vtarget d2 0 1.0

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def ratio_at_dvth(dvth_mv, w_um):
    w_nm = w_um * 1000.0
    vth0_ref = VTH0_REF_NOMINAL
    vth0_target = VTH0_REF_NOMINAL + dvth_mv / 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target, w_nm=w_nm)
    with open("_dvth_search.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_dvth_search.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_target / i_ref

# Binary search for the dvth (mV) where ratio == THRESHOLD
# Ratio decreases as dvth increases (bit=0 at dvth=0 has ratio~1.1, bit=1 at dvth=60 has ratio~0.22)
lo, hi = 0.0, 60.0
print(f"Searching for dvth where ratio = {THRESHOLD} (bisection, dvth in [0,60] mV)\n")
for i in range(20):
    mid = (lo + hi) / 2
    r = ratio_at_dvth(mid, W_UM)
    print(f"  dvth={mid:6.3f}mV -> ratio={r:.4f}")
    if r > THRESHOLD:
        lo = mid   # ratio needs to decrease -> need more dvth
    else:
        hi = mid

dvth_critical_mv = (lo + hi) / 2
print(f"\ndvth_critical (where ratio crosses threshold) = {dvth_critical_mv:.3f} mV")

# Phase 2's validated NBTI model: dVth(t) = 1.97 * t^0.3 (mV, t in years)
A, N = 1.97, 0.3
t_critical_years = (dvth_critical_mv / A) ** (1 / N)
print(f"\nUsing Phase 2's NBTI model dVth(t) = {A} * t^{N} (mV, t in years):")
print(f"Years of natural aging alone for an UNSTRESSED (bit=0) cell to reach")
print(f"the decision threshold: {t_critical_years:.2f} years")

print("\nCaveat: this is a deterministic center-value estimate (no mismatch spread).")
print("Real cells will vary around this due to Pelgrom mismatch (sigma=11.86mV at")
print("this design point) -- some cells will cross earlier, some later. A proper")
print("population estimate would add the mismatch sigma to dvth_critical and find")
print("the years-to-threshold distribution, not just the single center-value number.")