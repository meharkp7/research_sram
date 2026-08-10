import subprocess
import numpy as np

N_CHIPS = 15
BITS_PER_CHIP = 32
K_PAIRS = 4
VTH0_REF_NOMINAL = 0.46893
W_UM = 3.2
L_UM = 0.045
AVT_MV_UM = 4.5
SIGMA_LOCAL_MV = AVT_MV_UM / np.sqrt(W_UM * L_UM)

# Ratios of systematic:local sigma to test -- 0.5 (50%) already characterized in Phase 8.4,
# extending here to more aggressive ratios where any non-ideal leakage would be easiest to see.
SYSTEMATIC_RATIOS = [1.0, 2.0]

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

def run_trial(vth0_ref, vth0_target, w_um):
    w_nm = w_um * 1000.0
    netlist = NETLIST_TEMPLATE.format(vth0_ref=vth0_ref, vth0_target=vth0_target, w_nm=w_nm)
    with open("_sens_trial.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_sens_trial.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_ref, i_target

DVTH_BIT1_MV = 60.0

print(f"Local sigma: {SIGMA_LOCAL_MV:.2f}mV")
print(f"Testing systematic ratios: {SYSTEMATIC_RATIOS} (i.e. {[r*SIGMA_LOCAL_MV for r in SYSTEMATIC_RATIOS]} mV)\n")

all_results = {}

for ratio in SYSTEMATIC_RATIOS:
    sigma_systematic_mv = ratio * SIGMA_LOCAL_MV
    print(f"=== Systematic ratio {ratio}x ({sigma_systematic_mv:.2f}mV) ===")

    chip_idx_list, bit_pos_list, bit_val_list, ratio_list = [], [], [], []

    for chip_num in range(N_CHIPS):
        chip_systematic_offset_v = np.random.normal(0, sigma_systematic_mv) / 1000.0
        secret_bits = np.random.randint(0, 2, size=BITS_PER_CHIP)

        for bit_idx in range(BITS_PER_CHIP):
            dvth_v = (DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else 0.0) / 1000.0
            pair_ratios = []
            for pair in range(K_PAIRS):
                vth0_ref = (VTH0_REF_NOMINAL + chip_systematic_offset_v
                            + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0)
                vth0_target = (VTH0_REF_NOMINAL + dvth_v + chip_systematic_offset_v
                               + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0)
                i_ref, i_target = run_trial(vth0_ref, vth0_target, W_UM)
                pair_ratios.append(i_target / i_ref)
            chip_idx_list.append(chip_num)
            bit_pos_list.append(bit_idx)
            bit_val_list.append(secret_bits[bit_idx])
            ratio_list.append(np.mean(pair_ratios))

        if (chip_num + 1) % 5 == 0:
            print(f"  chip {chip_num+1}/{N_CHIPS} done")

    chip_idx_arr = np.array(chip_idx_list)
    bit_val_arr = np.array(bit_val_list)
    ratio_arr = np.array(ratio_list)

    # Leakage check, same methodology as the original Phase 8.4 test
    pop_mean_0 = ratio_arr[bit_val_arr == 0].mean()
    pop_mean_1 = ratio_arr[bit_val_arr == 1].mean()
    expected = np.where(bit_val_arr == 1, pop_mean_1, pop_mean_0)
    residual = ratio_arr - expected

    chip_mean_residual = np.array([residual[chip_idx_arr == c].mean() for c in range(N_CHIPS)])
    pop_residual_std = residual.std()
    expected_noise_only_std = pop_residual_std / np.sqrt(BITS_PER_CHIP)
    observed_std = chip_mean_residual.std()
    leakage_metric = observed_std / expected_noise_only_std

    print(f"  Observed/expected-if-cancelled ratio: {leakage_metric:.2f}x")
    all_results[ratio] = leakage_metric
    print()

print("=== Summary across all tested systematic:local ratios ===")
print(f"{'Ratio':>8} {'Leakage metric':>16}")
print(f"{'0.5x':>8} {'0.78x (Phase 8.4 original)':>26}")
for ratio, metric in all_results.items():
    print(f"{ratio:8.1f} {metric:16.2f}")

print("\nIf all values stay close to 1.0x (say, within roughly 0.5-2x), the 'no leakage'")
print("finding is robust across this range of systematic mismatch magnitudes -- not")
print("fragile to the specific 50% assumption originally tested.")