import subprocess
import numpy as np

np.random.seed(4242)

N_SCENARIOS = 20     # independent "chip deployed in some real-world condition" scenarios
BITS_PER_CHIP = 64
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
DVTH_BIT1_MV = 60.0
DVTH_BIT0_MV = 0.0
W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_LOCAL_MV = AVT_MV_UM / np.sqrt(W_UM * L_UM)          # ~11.86 mV
SIGMA_SYSTEMATIC_MV = 0.5 * SIGMA_LOCAL_MV                  # ~5.93 mV, Phase 8.4 assumption
MEAS_NOISE_REL_STD = 0.02

# Realistic joint deployment ranges
TEMP_RANGE_C = (0.0, 85.0)
VDD_RANGE_V = (0.9, 1.1)          # validated safe range, clear of the 1.3x GIDL cliff
AGING_YEARS_RANGE = (0.0, 10.0)   # time elapsed since tattoo writing, at read time

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
    with open("_compound_trial.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_compound_trial.sp"], capture_output=True, text=True)
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

# --- Phase A: calibration at NOMINAL conditions (25C, 1.0V, t=0 aging), on a fresh reference chip ---
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
print(f"Threshold established: {threshold:.4f}\n")

# --- Phase B: N_SCENARIOS independent chips, each under its OWN randomly drawn compound stress ---
print(f"=== Compound stress test: {N_SCENARIOS} scenarios, {BITS_PER_CHIP} bits each ===")
all_ber = []
scenario_conditions = []

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
        dvth_v = (base_dvth_mv + aging_drift_mv) / 1000.0  # natural aging adds on top of written state

        pair_ratios = []
        for pair in range(K_PAIRS):
            vth0_ref = (VTH0_REF_NOMINAL + chip_systematic_offset_mv / 1000.0
                        + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0)
            vth0_target = (VTH0_REF_NOMINAL + dvth_v + chip_systematic_offset_mv / 1000.0
                           + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0)
            i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM, temp_c, vdd)
            # Layer on readout noise at measurement time
            i_ref_noisy = i_ref * (1 + np.random.normal(0, MEAS_NOISE_REL_STD))
            i_target_noisy = i_target * (1 + np.random.normal(0, MEAS_NOISE_REL_STD))
            pair_ratios.append(i_target_noisy / i_ref_noisy)
        measured_ratio[bit_idx] = np.mean(pair_ratios)

    read_bits = (measured_ratio >= threshold).astype(int) if higher_is_1 else (measured_ratio <= threshold).astype(int)
    ber = np.mean(read_bits != secret_bits)
    all_ber.append(ber)
    scenario_conditions.append((temp_c, vdd, aging_years, chip_systematic_offset_mv, ber))

    print(f"  Scenario {scenario+1:2d}/{N_SCENARIOS}: T={temp_c:5.1f}C, VDD={vdd:.3f}V, "
          f"age={aging_years:4.1f}yr, sys_offset={chip_systematic_offset_mv:+.2f}mV -> BER={ber*100:.2f}%")

all_ber = np.array(all_ber)
print(f"\n=== Compound stress results across {N_SCENARIOS} scenarios ===")
print(f"Mean BER: {all_ber.mean()*100:.3f}%, std: {all_ber.std()*100:.3f}%, "
      f"min: {all_ber.min()*100:.3f}%, max: {all_ber.max()*100:.3f}%")
print(f"\nCompare to previously-established single-factor BERs:")
print(f"  Mismatch only (Segment 5.2):              0.78%")
print(f"  + temperature swing, isolated (5.6 fixed): 0.39-0.78%")
print(f"  + voltage swing, isolated (6.3):            0.78% (0 additional flips)")

np.savez("phase_compound_variation.npz",
         all_ber=all_ber, scenario_conditions=np.array(scenario_conditions, dtype=object),
         threshold=threshold, n_scenarios=N_SCENARIOS, bits_per_chip=BITS_PER_CHIP)
print("\nSaved to phase_compound_variation.npz")