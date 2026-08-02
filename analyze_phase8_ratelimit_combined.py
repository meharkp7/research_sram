import subprocess
import numpy as np

VTH0_REF_NOMINAL = 0.46893
W_UM = 3.2
AVT_MV_UM = 4.5
L_UM = 0.045
SIGMA_LOCAL_MV = (AVT_MV_UM / np.sqrt(W_UM * L_UM))

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
    with open("_cap_lookup.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_cap_lookup.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_target / i_ref

print("Building ratio(dvth) lookup table...")
dvth_grid = np.arange(-30, 101, 1.0)
ratio_grid = np.array([ratio_at_dvth(d) for d in dvth_grid])

def dvth_to_ratio(dvth):
    return np.interp(np.clip(dvth, dvth_grid.min(), dvth_grid.max()), dvth_grid, ratio_grid)

print(f"Done, {len(dvth_grid)} points.\n")

TARGET_DVTH_MV = 60.0
K_SIGMA = 3
RELIABILITY_STD = 0.0033
MATCH_WINDOW = K_SIGMA * np.sqrt(2) * RELIABILITY_STD
N_TRIALS = 500  # more trials for a smoother success-vs-cap curve

rng = np.random.default_rng(999)

# Collect the query-count needed for each trial (accept/reject, exhaustive grid search)
queries_needed = []
search_range = 150.0
step = max(2 * MATCH_WINDOW / (dvth_grid.max() - dvth_grid.min()) * search_range, 0.5)
grid = np.arange(0, search_range, step)

for trial in range(N_TRIALS):
    target_mismatch_mv = rng.normal(0, SIGMA_LOCAL_MV * np.sqrt(2))
    target_true_dvth = TARGET_DVTH_MV + target_mismatch_mv
    target_ratio = dvth_to_ratio(target_true_dvth)

    candidate_mismatch_mv = rng.normal(0, SIGMA_LOCAL_MV * np.sqrt(2))

    found_at = None
    for q, dvth_try in enumerate(grid):
        candidate_ratio = dvth_to_ratio(dvth_try + candidate_mismatch_mv)
        if abs(candidate_ratio - target_ratio) < MATCH_WINDOW:
            found_at = q + 1
            break
    queries_needed.append(found_at if found_at is not None else len(grid) + 1)

queries_needed = np.array(queries_needed)

print(f"Match window: {MATCH_WINDOW:.5f}, grid step: {step:.3f}mV, "
      f"grid size: {len(grid)} points\n")
print(f"{'Attempt cap':>12} {'Attack success rate within cap':>32}")
for cap in [5, 10, 20, 50, 100, 150, 200]:
    success_rate = np.mean(queries_needed <= cap)
    print(f"{cap:12d} {success_rate*100:31.1f}%")

print(f"\nFor reference: {N_TRIALS}-trial mean queries needed (uncapped): {queries_needed.mean():.1f}")
print(f"Median: {np.median(queries_needed):.1f}, 90th percentile: {np.percentile(queries_needed, 90):.1f}")

print("\nRecommendation: pick an attempt cap where success rate is acceptably low for the")
print("threat model, while remaining well above the number of attempts a LEGITIMATE user")
print("would ever need (a handful, given normal verification succeeds immediately).")