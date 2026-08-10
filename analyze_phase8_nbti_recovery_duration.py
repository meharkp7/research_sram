import subprocess
import numpy as np
import pandas as pd

# --- Stress-duration-dependent recoverable fraction model, REVISED ---
# The original design converted dvth -> implied stress years -> recoverable fraction, using
# Phase 2's power-law coefficient (1.97). That coefficient was found to be inconsistent with
# the same source document's own data-point labels (an ~8800x discrepancy in implied years) --
# NBTI rate constants are highly stress-condition-specific (voltage, temperature, oxide
# thickness), so there is no single universal literature value to substitute, unlike the
# Pelgrom mismatch coefficient case.
#
# FIX: skip the unreliable years-conversion entirely. Motivated by the same qualitative
# literature finding (recoverable fraction decreases with MORE accumulated damage), define
# the recoverable fraction as a direct function of accumulated dvth itself (0-70mV, the
# well-established, extensively-validated range used throughout this project) rather than
# an implied stress duration. This is more defensible: dvth is something we've directly
# simulated and trust; "years" requires an unresolved rate constant we do not have.
F_RECOVERABLE_MAX = 0.50   # near dvth=0 (freshly stressed / no accumulated damage)
F_RECOVERABLE_MIN = 0.15   # at maximum tested dvth (~70mV, severe) floor
DVTH_MAX_MV = 70.0         # upper bound of the severe category range

def f_recoverable(dvth_mv):
    frac = np.clip(dvth_mv / DVTH_MAX_MV, 0, 1)
    return F_RECOVERABLE_MAX - (F_RECOVERABLE_MAX - F_RECOVERABLE_MIN) * frac

# Print the resulting curve at reference points for transparency
print("dvth-dependent recoverable fraction model (avoids the unreliable years-conversion):")
for dvth in [0, 5, 15, 30, 45, 60, 70]:
    print(f"  dvth={dvth:5.1f}mV -> f_recoverable = {f_recoverable(dvth)*100:.1f}%")
print()

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
    with open("_recovery2_lookup.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_recovery2_lookup.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_target / i_ref

print("Building ratio(dvth) lookup table...")
dvth_grid = np.arange(-10, 76, 1.0)
ratio_grid = np.array([ratio_at_dvth(d) for d in dvth_grid])
print(f"Done, {len(dvth_grid)} points.\n")

def dvth_to_ratio(dvth):
    return np.interp(np.clip(dvth, dvth_grid.min(), dvth_grid.max()), dvth_grid, ratio_grid)

FINE_THRESHOLDS = [0.3384, 0.6194, 0.9393]
FINE_ORDER = ["severe", "moderate", "light", "fresh"]

def classify_fine(ratio):
    for cat, t in zip(FINE_ORDER[:-1], FINE_THRESHOLDS):
        if ratio < t:
            return cat
    return FINE_ORDER[-1]

def is_suspect(cat):
    return cat in ("moderate", "severe")

# --- Reuse real Phase 4 chip data ---
df = pd.read_pickle("unified_leakage_dataset.pkl")
task_a = df[df["application"] == "counterfeit_detection"].copy()
real_chips = task_a[task_a["ground_truth_label"].isin(["moderate", "severe"])].copy()
print(f"Testing on {len(real_chips)} real 'moderate'/'severe' chips\n")

fine_evasions = 0
binary_evasions = 0
results = []

for _, row in real_chips.iterrows():
    true_dvth = row["dvth_effective_mv"]
    true_cat = row["ground_truth_label"]

    f_recov = f_recoverable(true_dvth)  # depends on dvth directly, not an unreliable years-conversion

    post_recovery_dvth = true_dvth * (1 - f_recov)
    post_recovery_ratio = dvth_to_ratio(post_recovery_dvth)
    post_recovery_cat = classify_fine(post_recovery_ratio)

    if post_recovery_cat != true_cat:
        fine_evasions += 1
    if is_suspect(true_cat) and not is_suspect(post_recovery_cat):
        binary_evasions += 1

    results.append((true_cat, true_dvth, f_recov, post_recovery_cat))

print(f"=== Results, stress-duration-dependent recovery model ===")
print(f"Fine-grained evasions: {fine_evasions}/{len(real_chips)} ({fine_evasions/len(real_chips)*100:.1f}%)")
print(f"Binary-flag evasions:  {binary_evasions}/{len(real_chips)} ({binary_evasions/len(real_chips)*100:.1f}%)")

print(f"\nBreakdown by original category:")
for cat in ["moderate", "severe"]:
    cat_results = [r for r in results if r[0] == cat]
    if not cat_results:
        continue
    cat_binary_evasions = sum(1 for r in cat_results if not is_suspect(r[3]))
    mean_f_recov = np.mean([r[2] for r in cat_results])
    print(f"  {cat:10s}: n={len(cat_results):3d}, mean f_recoverable={mean_f_recov*100:.1f}%, "
          f"binary evasions={cat_binary_evasions}/{len(cat_results)} "
          f"({cat_binary_evasions/len(cat_results)*100:.1f}%)")

print(f"\nCompare to the original FLAT-fraction results (Segment 8.6):")
print(f"  At 20% flat:  3.75% binary evasion")
print(f"  At 30% flat: 11.25% binary evasion")
print(f"This dvth-dependent model should show LOWER evasion for 'severe' chips specifically")
print(f"(more accumulated damage -> lower recoverable fraction) and HIGHER evasion for")
print(f"'moderate' chips (less accumulated damage -> higher recoverable fraction) -- a more")
print(f"physically nuanced picture than the uniform-fraction sensitivity sweep provided.")