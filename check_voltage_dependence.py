import subprocess
import numpy as np

VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
W_UM = 3.2
VDD_SWEEP = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4]  # V -- adjust range if your node's nominal VDD differs

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w={w_nm:.0f}n
Mtarget d2 0 0 0 nmos_target l=45n w={w_nm:.0f}n

Vref    d1 0 {vdd}
Vtarget d2 0 {vdd}

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def run_trial(vth0_ref, vth0_target, w_um, vdd):
    w_nm = w_um * 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target,
                                       w_nm=w_nm, vdd=vdd)
    with open("_vdd_check.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_vdd_check.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    if i_ref is None or i_target is None:
        raise RuntimeError(f"Parse failure. Raw output:\n{result.stdout}")
    return i_ref, i_target

vth0_ref = VTH0_REF_NOMINAL
vth0_target = VTH0_REF_NOMINAL + DVTH_BIT1_MV / 1000.0

print(f"{'VDD (V)':>10} {'I_ref (nA)':>14} {'I_target (nA)':>14} {'ratio':>10}")
results = []
for vdd in VDD_SWEEP:
    i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM, vdd)
    ratio = i_target / i_ref
    results.append((vdd, i_ref, i_target, ratio))
    print(f"{vdd:10.2f} {i_ref*1e9:14.4f} {i_target*1e9:14.4f} {ratio:10.4f}")

i_refs = np.array([r[1] for r in results])
i_targets = np.array([r[2] for r in results])
ratios = np.array([r[3] for r in results])

def spread_pct(x):
    return 100 * (x.max() - x.min()) / x.mean()

print(f"\nI_ref relative spread across VDD: {spread_pct(i_refs):.1f}%")
print(f"I_target relative spread across VDD: {spread_pct(i_targets):.1f}%")
print(f"Ratio relative spread across VDD: {spread_pct(ratios):.1f}%")
print("\nIf I_ref/I_target vary substantially with VDD (real DIBL), check whether the")
print("ratio spread is much smaller -- that's the cancellation property. If currents")
print("barely move with VDD, the model has weak DIBL and this dimension may need a")
print("stated behavioral addition instead (same caveat pattern as your other Enhancement 3 notes).")