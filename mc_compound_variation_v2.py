import subprocess
import numpy as np

np.random.seed(8181)  # new seed -- independent replication, not a re-run of the same 20 scenarios

N_SCENARIOS = 40
BITS_PER_CHIP = 32   # reduced from 64 to manage total runtime while doubling scenario count --
                      # between-scenario variance (the tail) matters more here than per-scenario precision
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
DVTH_BIT0_MV = 0.0
W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_LOCAL_MV = AVT_MV_UM / np.sqrt(W_UM * L_UM)
SIGMA_SYSTEMATIC_MV = 0.5 * SIGMA_LOCAL_MV
MEAS_NOISE_REL_STD = 0.02

TEMP_RANGE_C = (0.0, 85.0)
VDD_RANGE_V = (0.9, 1.1)
AGING_YEARS_RANGE = (0.0, 10.0)

NETLIST_TEMPLATE = """.include ./45nm_HP_nmos_ref_param.pm
.include ./45nm_HP_nmos_target_param.pm

.param vth0_ref = {vth0_ref:.6f}
.param vth0_target = {vth0_target:.6f}

Mref    d1 0 0 0 nmos_ref    l=45n w={w_nm:.0f}n
Mtarget d2 0 0 0 nmos_target l=45n w={w_nm:.0f}n

Vref    d1 0 {vdd}
Vtarget d2 0 {vdd}

.temp {temp_c}

.control
op
print i(Vref) i(Vtarget)
.endc
.end
"""

def run_trial(vth0_ref, vth0_target, w_um, temp_c, vdd):
    w_nm = w_um * 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target,
                                       w_nm=w_nm, temp_c=temp_c, vdd=vdd)
    with open("_compound2_trial.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_compound2_trial.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_ref, i_target

print("=== Calibration pass (nominal: 25C, 1.0V, t=0) ===")
calib_secret_bits = np.random.randint(0, 2, size=BITS_PER_CHIP)
calib_ratio = np.zeros(BITS_PER_CHIP)
for bit_idx in range(BITS_PER_CHIP):
    dvth_v = (DVTH_BIT1_MV if calib_secret_bits[bit_idx] == 1 else DVTH_BIT0_MV) / 1000.0
    pair_ratios = []
    for pair in range(K_PAIRS):
        vth0_ref = VTH0_REF_NOMINAL + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0
        vth0_target = VTH0_REF_NOMINAL + dvth_v + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0
        i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM, 25.0, 1.0)
        pair_ratios.append(i_target / i_ref)
    calib_ratio[bit_idx] = np.mean(pair_ratios)

mean1 = calib_ratio[calib_secret_bits == 1].mean()
mean0 = calib_ratio[calib_secret_bits == 0].mean()
threshold = (mean1 + mean0) / 2
higher_is_1 = mean1 > mean0
print(f"Threshold: {threshold:.4f}\n")

print(f"=== Compound stress test: {N_SCENARIOS} scenarios, {BITS_PER_CHIP} bits each ===")
all_ber = []
all_conditions = []

for scenario in range(N_SCENARIOS):
    chip_systematic_offset_mv = np.random.normal(0, SIGMA_SYSTEMATIC_MV)
    temp_c = np.random.uniform(*TEMP_RANGE_C)
    vdd = np.random.uniform(*VDD_RANGE_V)
    aging_years = np.random.uniform(*AGING_YEARS_RANGE)
    aging_drift_mv = 1.97 * (aging_years ** 0.3) if aging_years > 0 else 0.0

    secret_bits = np.random.randint(0, 2, size=BITS_PER_CHIP)
    measured_ratio = np.zeros(BITS_PER_CHIP)

    for bit_idx in range(BITS_PER_CHIP):
        base_dvth_mv = DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else DVTH_BIT0_MV
        dvth_v = (base_dvth_mv + aging_drift_mv) / 1000.0
        pair_ratios = []
        for pair in range(K_PAIRS):
            vth0_ref = (VTH0_REF_NOMINAL + chip_systematic_offset_mv / 1000.0
                        + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0)
            vth0_target = (VTH0_REF_NOMINAL + dvth_v + chip_systematic_offset_mv / 1000.0
                           + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0)
            i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM, temp_c, vdd)
            i_ref_noisy = i_ref * (1 + np.random.normal(0, MEAS_NOISE_REL_STD))
            i_target_noisy = i_target * (1 + np.random.normal(0, MEAS_NOISE_REL_STD))
            pair_ratios.append(i_target_noisy / i_ref_noisy)
        measured_ratio[bit_idx] = np.mean(pair_ratios)

    read_bits = (measured_ratio >= threshold).astype(int) if higher_is_1 else (measured_ratio <= threshold).astype(int)
    ber = np.mean(read_bits != secret_bits)
    all_ber.append(ber)
    all_conditions.append((temp_c, vdd, aging_years, chip_systematic_offset_mv))

    if (scenario + 1) % 10 == 0:
        print(f"  {scenario+1}/{N_SCENARIOS} scenarios done")

all_ber = np.array(all_ber)

print(f"\n=== Raw results, {N_SCENARIOS} scenarios ===")
print(f"Mean BER: {all_ber.mean()*100:.3f}%, std: {all_ber.std()*100:.3f}%, "
      f"min: {all_ber.min()*100:.3f}%, max: {all_ber.max()*100:.3f}%")

# --- Bootstrap CI on mean BER ---
rng = np.random.default_rng(999)
N_BOOTSTRAP = 10000
bootstrap_means = np.array([
    rng.choice(all_ber, size=N_SCENARIOS, replace=True).mean()
    for _ in range(N_BOOTSTRAP)
])
ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"\nBootstrap 95% CI on mean BER: [{ci_low*100:.3f}%, {ci_high*100:.3f}%]")

# --- Empirical tail estimate ---
print(f"\nEmpirical percentiles of per-scenario BER:")
for p in [50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(all_ber, p)*100:.3f}%")

np.savez("phase_compound_variation_v2.npz",
         all_ber=all_ber, all_conditions=np.array(all_conditions),
         threshold=threshold, bootstrap_means=bootstrap_means,
         ci_low=ci_low, ci_high=ci_high)
print("\nSaved to phase_compound_variation_v2.npz")