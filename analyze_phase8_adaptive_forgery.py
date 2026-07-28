import subprocess
import numpy as np

VTH0_REF_NOMINAL = 0.46893
W_UM = 3.2
AVT_MV_UM = 4.5
L_UM = 0.045
SIGMA_LOCAL_MV = (AVT_MV_UM / np.sqrt(W_UM * L_UM))  # ~11.86 mV

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
    with open("_forge_lookup.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_forge_lookup.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_target / i_ref

print("Building ratio(dvth) lookup table for the forgery-attack simulation...")
dvth_grid = np.arange(-30, 61, 1.0)
ratio_grid = np.array([ratio_at_dvth(d) for d in dvth_grid])

def dvth_to_ratio_correct(dvth):
    return np.interp(dvth, dvth_grid, ratio_grid)

print(f"Lookup table built, {len(dvth_grid)} points from {dvth_grid.min()}mV to {dvth_grid.max()}mV\n")

# ============ Adaptive forgery attack simulation ============
# Target: a genuine chip stressed to TARGET_DVTH_MV=60mV (bit=1 / severe-style target),
# with its OWN random mismatch. Attacker knows the target's MEASURED ratio (assumed exact,
# given how tight reliability_std is) and tries to converge their own candidate to match it,
# via bisection on their candidate's INTENDED stress level. The candidate has its own unknown
# (to attacker) local mismatch offset.

TARGET_DVTH_MV = 60.0
N_TRIALS = 200
BISECTION_TOLERANCE_MV = 0.5  # attacker's assumed achievable stress-control precision
MAX_ITERATIONS = 30

rng = np.random.default_rng(2026)

convergence_iters = []
final_ratio_errors = []
stuck_count = 0  # candidates whose baseline mismatch already exceeds the target (can't converge forward)

for trial in range(N_TRIALS):
    # True target: genuine chip with its own mismatch
    target_mismatch_mv = rng.normal(0, SIGMA_LOCAL_MV * np.sqrt(2))  # combined ref+target mismatch, approx
    target_true_dvth = TARGET_DVTH_MV + target_mismatch_mv
    target_ratio = dvth_to_ratio_correct(np.clip(target_true_dvth, dvth_grid.min(), dvth_grid.max()))

    # Attacker's candidate: own unknown mismatch, starts at dvth_intended=0 (unstressed)
    candidate_mismatch_mv = rng.normal(0, SIGMA_LOCAL_MV * np.sqrt(2))

    # Bisection search on dvth_intended (attacker's controllable stress amount, >=0, monotonically increasing only)
    lo, hi = 0.0, 100.0  # attacker's achievable stress range
    candidate_ratio_at_lo = dvth_to_ratio_correct(np.clip(lo + candidate_mismatch_mv, dvth_grid.min(), dvth_grid.max()))
    candidate_ratio_at_hi = dvth_to_ratio_correct(np.clip(hi + candidate_mismatch_mv, dvth_grid.min(), dvth_grid.max()))

    if candidate_ratio_at_lo <= target_ratio:
        # Candidate's baseline (even unstressed) is already "more aged" than target -- can't converge forward
        stuck_count += 1
        continue

    iters = 0
    for i in range(MAX_ITERATIONS):
        mid = (lo + hi) / 2
        candidate_ratio = dvth_to_ratio_correct(np.clip(mid + candidate_mismatch_mv, dvth_grid.min(), dvth_grid.max()))
        iters += 1
        if abs(candidate_ratio - target_ratio) < 1e-5 or (hi - lo) < BISECTION_TOLERANCE_MV:
            break
        if candidate_ratio > target_ratio:
            lo = mid  # need more stress
        else:
            hi = mid

    final_ratio = dvth_to_ratio_correct(np.clip(mid + candidate_mismatch_mv, dvth_grid.min(), dvth_grid.max()))
    convergence_iters.append(iters)
    final_ratio_errors.append(abs(final_ratio - target_ratio))

convergence_iters = np.array(convergence_iters)
final_ratio_errors = np.array(final_ratio_errors)

print(f"=== Adaptive forgery attack simulation ({N_TRIALS} trials) ===")
print(f"Trials where candidate got 'stuck' (baseline mismatch already past target): "
      f"{stuck_count}/{N_TRIALS} ({stuck_count/N_TRIALS*100:.1f}%)")
print(f"Successful convergence attempts: {len(convergence_iters)}/{N_TRIALS}")
print(f"\nAmong successful attempts:")
print(f"  Mean iterations to converge: {convergence_iters.mean():.1f} (max: {convergence_iters.max()})")
print(f"  Mean final |ratio error|: {final_ratio_errors.mean():.5f}")
print(f"  Compare to reliability_std (Segment 5.3): 0.0033")
print(f"  Compare to uniqueness_std (Segment 5.3):  0.0465")

reliability_std = 0.0033
frac_within_reliability = np.mean(final_ratio_errors < reliability_std)
print(f"\nFraction of successful forgeries landing within 1 reliability-sigma of the target: "
      f"{frac_within_reliability*100:.1f}%")
print("\nIf this fraction is high, the adaptive attack achieves forgery precision comparable to")
print("simply re-measuring the SAME genuine chip twice -- i.e., it defeats Game G3's cloning")
print("resistance almost completely, for any target the attacker can physically access.")