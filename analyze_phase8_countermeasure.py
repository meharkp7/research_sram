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
    with open("_countermeasure_lookup.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_countermeasure_lookup.sp"], capture_output=True, text=True)
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
K_SIGMA = 3  # matching window width, in reliability-sigmas (same as Segment 5.5's k=3)
RELIABILITY_STD = 0.0033
MATCH_WINDOW = K_SIGMA * np.sqrt(2) * RELIABILITY_STD  # same definition as Segment 8.6/analyze_phase5_auth_tradeoff
N_TRIALS = 200
MAX_QUERIES = 200  # attacker's query budget to the verifier per attempt

rng = np.random.default_rng(555)

def run_scenario(feedback_type):
    """feedback_type: 'full', 'directional', 'accept_reject'"""
    successes = 0
    queries_used = []

    for trial in range(N_TRIALS):
        target_mismatch_mv = rng.normal(0, SIGMA_LOCAL_MV * np.sqrt(2))
        target_true_dvth = TARGET_DVTH_MV + target_mismatch_mv
        target_ratio = dvth_to_ratio(target_true_dvth)

        candidate_mismatch_mv = rng.normal(0, SIGMA_LOCAL_MV * np.sqrt(2))

        if feedback_type in ("full", "directional"):
            # Bisection search -- works identically whether feedback is the exact value
            # (attacker computes direction themselves) or just the direction bit from the verifier.
            lo, hi = 0.0, 150.0
            found = False
            for q in range(MAX_QUERIES):
                mid = (lo + hi) / 2
                candidate_ratio = dvth_to_ratio(mid + candidate_mismatch_mv)
                if abs(candidate_ratio - target_ratio) < MATCH_WINDOW:
                    found = True
                    queries_used.append(q + 1)
                    break
                if candidate_ratio > target_ratio:
                    lo = mid
                else:
                    hi = mid
            if found:
                successes += 1
            else:
                queries_used.append(MAX_QUERIES)

        elif feedback_type == "accept_reject":
            # No gradient/direction info at all -- attacker only learns PASS/FAIL.
            # Best strategy without directional info: systematic grid search across the
            # achievable stress range, since no better-than-random strategy exists.
            search_range = 150.0
            step = 2 * MATCH_WINDOW / (dvth_grid.max() - dvth_grid.min()) * search_range  # rough grid step
            step = max(step, 0.5)
            grid = np.arange(0, search_range, step)
            found = False
            for q, dvth_try in enumerate(grid):
                if q >= MAX_QUERIES:
                    break
                candidate_ratio = dvth_to_ratio(dvth_try + candidate_mismatch_mv)
                if abs(candidate_ratio - target_ratio) < MATCH_WINDOW:
                    found = True
                    queries_used.append(q + 1)
                    break
            if found:
                successes += 1
            else:
                queries_used.append(MAX_QUERIES)

    return successes, np.array(queries_used)

print(f"Match window (k={K_SIGMA} sigma): {MATCH_WINDOW:.5f}\n")
print(f"{'Feedback model':20s} {'Success rate':>14} {'Mean queries':>14} {'Median queries':>16}")
for feedback in ["full", "directional", "accept_reject"]:
    successes, queries = run_scenario(feedback)
    print(f"{feedback:20s} {successes/N_TRIALS*100:13.1f}% {queries.mean():14.1f} {np.median(queries):16.1f}")

print("\nInterpretation:")
print("- 'full' and 'directional' should perform SIMILARLY (bisection only needs direction,")
print("  not magnitude) -- confirming that hiding exact values but leaking direction is NOT")
print("  sufficient as a countermeasure.")
print("- 'accept_reject' (pure pass/fail, zero directional leakage) should need dramatically")
print("  more queries, closer to the passive FAR-based search cost from Segment 8.3 -- this")
print("  is the countermeasure that actually works, and it requires a genuinely constant-time,")
print("  non-data-dependent verifier implementation (Segment 8.5's requirement, now shown to")
print("  be load-bearing rather than a side concern).")