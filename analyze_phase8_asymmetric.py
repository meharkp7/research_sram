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
SIGMA_SYSTEMATIC_MV = 0.5 * SIGMA_LOCAL_MV  # same 50% assumption as the original test
DVTH_BIT1_MV = 60.0

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
    with open("_asym_trial.sp", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "_asym_trial.sp"], capture_output=True, text=True)
    i_ref = i_target = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("i(vref)"):
            i_ref = abs(float(line.split("=")[1].strip()))
        elif line.startswith("i(vtarget)"):
            i_target = abs(float(line.split("=")[1].strip()))
    return i_ref, i_target

print(f"Local sigma: {SIGMA_LOCAL_MV:.2f}mV, systematic sigma: {SIGMA_SYSTEMATIC_MV:.2f}mV")
print(f"ASYMMETRIC model: ref and target get INDEPENDENTLY drawn systematic offsets")
print(f"(representing poor common-centroid layout matching)\n")

chip_idx_list, bit_pos_list, bit_val_list, ratio_list = [], [], [], []

for chip_num in range(N_CHIPS):
    # Two SEPARATE, independent systematic offsets -- this is the key difference from the
    # original test, which used ONE shared offset applied identically to both transistors.
    ref_systematic_offset_v = np.random.normal(0, SIGMA_SYSTEMATIC_MV) / 1000.0
    target_systematic_offset_v = np.random.normal(0, SIGMA_SYSTEMATIC_MV) / 1000.0

    secret_bits = np.random.randint(0, 2, size=BITS_PER_CHIP)

    for bit_idx in range(BITS_PER_CHIP):
        dvth_v = (DVTH_BIT1_MV if secret_bits[bit_idx] == 1 else 0.0) / 1000.0
        pair_ratios = []
        for pair in range(K_PAIRS):
            vth0_ref = (VTH0_REF_NOMINAL + ref_systematic_offset_v
                        + np.random.normal(0, SIGMA_LOCAL_MV) / 1000.0)
            vth0_target = (VTH0_REF_NOMINAL + dvth_v + target_systematic_offset_v
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
bit_pos_arr = np.array(bit_pos_list)
bit_val_arr = np.array(bit_val_list)
ratio_arr = np.array(ratio_list)

np.savez("phase8_asymmetric_correlation.npz",
         chip_idx=chip_idx_arr, bit_position=bit_pos_arr, bit_value=bit_val_arr, ratio=ratio_arr,
         n_chips=N_CHIPS, bits_per_chip=BITS_PER_CHIP, sigma_systematic_mv=SIGMA_SYSTEMATIC_MV)

# --- Same leakage check and modeling-attack test as the original Phase 8.4 analysis ---
pop_mean_0 = ratio_arr[bit_val_arr == 0].mean()
pop_mean_1 = ratio_arr[bit_val_arr == 1].mean()
print(f"\nPopulation means: bit=0: {pop_mean_0:.4f}, bit=1: {pop_mean_1:.4f}")

expected = np.where(bit_val_arr == 1, pop_mean_1, pop_mean_0)
residual = ratio_arr - expected
chip_mean_residual = np.array([residual[chip_idx_arr == c].mean() for c in range(N_CHIPS)])
pop_residual_std = residual.std()
expected_noise_only_std = pop_residual_std / np.sqrt(BITS_PER_CHIP)
observed_std = chip_mean_residual.std()
leakage_metric = observed_std / expected_noise_only_std
print(f"Observed/expected-if-cancelled ratio: {leakage_metric:.2f}x")
if leakage_metric > 1.5:
    print("-> LEAKAGE DETECTED: asymmetric systematic mismatch does NOT fully cancel in the")
    print("   ratio, unlike the symmetric (well-matched-layout) case. This is a real, distinct")
    print("   finding -- common-mode rejection depends on ref/target sharing correlated variation,")
    print("   which requires good layout, not something the ratio metric provides automatically.")
else:
    print("-> Still close to full cancellation even in the asymmetric case.")

# Modeling attack: same M-observed-cells-predict-held-out-cells test as before
M_OBSERVED = 10
naive_threshold = (pop_mean_0 + pop_mean_1) / 2
naive_higher_is_1 = pop_mean_1 > pop_mean_0
rng = np.random.default_rng(42)

naive_accs, attack_accs = [], []
for chip in range(N_CHIPS):
    mask = chip_idx_arr == chip
    chip_ratio = ratio_arr[mask]
    chip_bitval = bit_val_arr[mask]
    chip_residual = residual[mask]
    idx = rng.permutation(len(chip_ratio))
    obs_idx, held_idx = idx[:M_OBSERVED], idx[M_OBSERVED:]

    pred_naive = (chip_ratio[held_idx] >= naive_threshold).astype(int) if naive_higher_is_1 \
                 else (chip_ratio[held_idx] <= naive_threshold).astype(int)
    naive_accs.append(np.mean(pred_naive == chip_bitval[held_idx]))

    est_bias = chip_residual[obs_idx].mean()
    adj_threshold = naive_threshold + est_bias
    pred_attack = (chip_ratio[held_idx] >= adj_threshold).astype(int) if naive_higher_is_1 \
                  else (chip_ratio[held_idx] <= adj_threshold).astype(int)
    attack_accs.append(np.mean(pred_attack == chip_bitval[held_idx]))

naive_accs, attack_accs = np.array(naive_accs), np.array(attack_accs)
print(f"\n=== Modeling attack test (asymmetric case, {M_OBSERVED} observed / "
      f"{BITS_PER_CHIP-M_OBSERVED} held-out) ===")
print(f"Naive accuracy:  {naive_accs.mean()*100:.2f}% (std: {naive_accs.std()*100:.2f}%)")
print(f"Attack accuracy: {attack_accs.mean()*100:.2f}% (std: {attack_accs.std()*100:.2f}%)")
gap = attack_accs.mean() - naive_accs.mean()
print(f"Gap: {gap*100:+.2f} points")
if gap > max(naive_accs.std(), attack_accs.std()):
    print("-> Meaningful gap: the asymmetric-correlation modeling attack DOES provide a real")
    print("   advantage here, unlike the symmetric case. This is a genuine, distinct vulnerability")
    print("   tied specifically to layout quality (ref/target matching), not the ratio metric itself.")
else:
    print("-> Gap within noise -- no meaningful exploitable advantage even in the asymmetric case.")