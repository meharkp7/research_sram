import subprocess
import numpy as np

VTH0_REF_NOMINAL = 0.46893
W_UM = 3.2
THRESHOLD = 0.6614

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
    with open("_dvth_lookup.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_dvth_lookup.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_target / i_ref

# Build a lookup table covering negative dvth too (since mismatch can push either direction)
dvth_grid = np.arange(-30, 61, 2.0)
print(f"Building ratio(dvth) lookup table, {len(dvth_grid)} points...")
ratio_grid = np.array([ratio_at_dvth(d, W_UM) for d in dvth_grid])

# ratio is monotonically decreasing in dvth -- for np.interp we need x increasing,
# so flip to use ratio (decreasing -> use -ratio as x, or reverse arrays)
order = np.argsort(ratio_grid)
ratio_sorted = ratio_grid[order]
dvth_sorted = dvth_grid[order]

def invert_ratio_to_dvth(ratio):
    return np.interp(ratio, ratio_sorted, dvth_sorted)

dvth_critical_mv = invert_ratio_to_dvth(THRESHOLD)
print(f"dvth_critical (sanity check vs Segment 5.7 bisection): {dvth_critical_mv:.3f} mV\n")

# --- Load the REAL simulated bit=0 cells from Segment 5.2 ---
data = np.load("phase5_256bit_readback.npz")
secret_bits = data["secret_bits"]
measured_ratio = data["measured_ratio"]

bit0_ratios = measured_ratio[secret_bits == 0]
print(f"n bit=0 cells: {len(bit0_ratios)}")

# Invert each real cell's measured ratio to its actual mismatch-induced dvth offset
effective_dvth = invert_ratio_to_dvth(bit0_ratios)
print(f"Effective dvth (mismatch only, t=0) across bit=0 cells: "
      f"mean={effective_dvth.mean():.3f}mV, std={effective_dvth.std():.3f}mV, "
      f"min={effective_dvth.min():.3f}mV, max={effective_dvth.max():.3f}mV")

n_already_over = np.sum(effective_dvth >= dvth_critical_mv)
print(f"Cells already past threshold at t=0 (pure mismatch, matches Segment 5.2's raw errors): "
      f"{n_already_over}")

# --- Apply Phase 2's NBTI aging model, find years-to-threshold per cell ---
A, N = 1.97, 0.3
remaining_mv = dvth_critical_mv - effective_dvth  # how much more drift each cell can tolerate
years_to_fail = np.full(len(effective_dvth), np.inf)
still_ok = remaining_mv > 0
years_to_fail[still_ok] = (remaining_mv[still_ok] / A) ** (1 / N)

finite_years = years_to_fail[np.isfinite(years_to_fail)]
print(f"\nYears-to-failure distribution across {len(effective_dvth)} real bit=0 cells:")
print(f"  Already failed (t=0): {np.sum(~still_ok)}")
print(f"  Min years (earliest future failure): {finite_years.min():.2f}" if len(finite_years) else "  N/A")
print(f"  Median years: {np.median(finite_years):.2f}" if len(finite_years) else "  N/A")
print(f"  5th percentile years: {np.percentile(finite_years, 5):.2f}" if len(finite_years) else "  N/A")

for horizon in [1, 5, 10, 20, 50]:
    n_fail_by = np.sum(years_to_fail <= horizon)
    print(f"  Cells expected to fail within {horizon:3d} years: {n_fail_by}/{len(effective_dvth)}")