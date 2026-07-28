import numpy as np
import pandas as pd

df = pd.read_pickle("unified_leakage_dataset.pkl")
task_a = df[df["application"] == "counterfeit_detection"].copy()

CATS = ["fresh", "light", "moderate", "severe"]
CAT_RANGES_MV = {"fresh": (0, 0), "light": (5, 20), "moderate": (20, 45), "severe": (45, 70)}

# Grid-searched thresholds from Segment 7.2 (fine-grained, ratio units)
FINE_THRESHOLDS = [0.3384, 0.6194, 0.9393]  # ascending: severe|moderate|light|fresh boundaries
FINE_ORDER = ["severe", "moderate", "light", "fresh"]

# Binary "suspect" grouping (moderate+severe vs fresh+light) -- reconstruct via same ratio data
def classify_fine(ratio):
    for cat, t in zip(FINE_ORDER[:-1], FINE_THRESHOLDS):
        if ratio < t:
            return cat
    return FINE_ORDER[-1]

def is_suspect(cat):
    return cat in ("moderate", "severe")

# We need the ratio(dvth) relationship to apply recovery (shift dvth, recompute ratio).
# Reuse the lookup table approach -- build it once, fast (reuses same physics as everywhere else).
import subprocess

VTH0_REF_NOMINAL = 0.46893
W_UM = 3.2
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

def ratio_at_dvth(dvth_mv, w_um=W_UM):
    w_nm = w_um * 1000.0
    vth0_ref = VTH0_REF_NOMINAL
    vth0_target = VTH0_REF_NOMINAL + dvth_mv / 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target, w_nm=w_nm)
    with open("_recovery_lookup.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_recovery_lookup.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_target / i_ref

print("Building ratio(dvth) lookup table for recovery-attack simulation...")
dvth_grid = np.arange(-10, 76, 1.0)
ratio_grid = np.array([ratio_at_dvth(d) for d in dvth_grid])
print(f"Done, {len(dvth_grid)} points.\n")

def dvth_to_ratio(dvth):
    return np.interp(np.clip(dvth, dvth_grid.min(), dvth_grid.max()), dvth_grid, ratio_grid)

# Use REAL simulated dvth_effective_mv from Phase 4's validated chip data
real_chips = task_a[["dvth_effective_mv", "ground_truth_label"]].copy()
real_chips = real_chips[real_chips["ground_truth_label"].isin(["moderate", "severe"])]
print(f"Testing recovery attack on {len(real_chips)} real 'moderate'/'severe' chips "
      f"from Segment 7.2's validated dataset\n")

RECOVERABLE_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.65]

print(f"{'f_recoverable':>14} {'fine-grained evasions':>22} {'binary-flag evasions':>22}")
for f_recov in RECOVERABLE_FRACTIONS:
    fine_evasions = 0
    binary_evasions = 0
    for _, row in real_chips.iterrows():
        true_dvth = row["dvth_effective_mv"]
        true_cat = row["ground_truth_label"]

        post_recovery_dvth = true_dvth * (1 - f_recov)
        post_recovery_ratio = dvth_to_ratio(post_recovery_dvth)
        post_recovery_cat = classify_fine(post_recovery_ratio)

        if post_recovery_cat != true_cat:
            fine_evasions += 1
        if is_suspect(true_cat) and not is_suspect(post_recovery_cat):
            binary_evasions += 1

    print(f"{f_recov*100:13.0f}% {fine_evasions:19d}/{len(real_chips)} "
          f"{binary_evasions:19d}/{len(real_chips)}")

print("\n'Fine-grained evasions': chip's reported CATEGORY changes after recovery")
print("'Binary-flag evasions': chip moves from suspect (moderate/severe) to NOT-suspect")
print("(fresh/light) -- this is the safety-critical evasion that actually matters for")
print("counterfeit detection deployment, per Phase 4's own established binary framing.")