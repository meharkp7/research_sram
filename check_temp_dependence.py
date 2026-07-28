import subprocess
import numpy as np

VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
W_UM = 3.2
TEMPS_C = [0, 25, 50, 75, 100]

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w={w_nm:.0f}n
Mtarget d2 0 0 0 nmos_target l=45n w={w_nm:.0f}n

Vref    d1 0 1.0
Vtarget d2 0 1.0

.temp {temp_c}

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def run_trial(vth0_ref, vth0_target, w_um, temp_c):
    w_nm = w_um * 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target,
                                       w_nm=w_nm, temp_c=temp_c)
    with open("_temp_check.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_temp_check.sp"], capture_output=True, text=True)
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

# Fixed, deterministic Vth (no mismatch draw) -- isolating pure temperature effect
vth0_ref = VTH0_REF_NOMINAL
vth0_target = VTH0_REF_NOMINAL + DVTH_BIT1_MV / 1000.0

print(f"{'Temp (C)':>10} {'I_ref (nA)':>14} {'I_target (nA)':>14} {'ratio':>10}")
results = []
for t in TEMPS_C:
    i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM, t)
    ratio = i_target / i_ref
    results.append((t, i_ref, i_target, ratio))
    print(f"{t:10d} {i_ref*1e9:14.4f} {i_target*1e9:14.4f} {ratio:10.4f}")

i_refs = np.array([r[1] for r in results])
i_targets = np.array([r[2] for r in results])
ratios = np.array([r[3] for r in results])

def spread_pct(x):
    return 100 * (x.max() - x.min()) / x.mean()

print(f"\nI_ref relative spread across temperature: {spread_pct(i_refs):.1f}%")
print(f"I_target relative spread across temperature: {spread_pct(i_targets):.1f}%")
print(f"Ratio relative spread across temperature: {spread_pct(ratios):.1f}%")
print("\nIf I_ref/I_target barely change across temperature (<~5% spread), the .pm model")
print("likely has weak/no real temperature dependence and this test isn't meaningful --")
print("tell me and we'll either add an explicit temperature-dependent behavioral term")
print("or treat Segment 3.2's original 44.6% figure as the best available estimate.")
print("If they DO change substantially, check whether the ratio spread is much smaller")
print("than the I_ref/I_target spread -- that's the cancellation property we're testing.")