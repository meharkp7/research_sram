import numpy as np

K_PAIRS = 4
W_NM = 3200
DATASET_FILE = f"phase5_tattoo_uniqueness_K{K_PAIRS}_w{W_NM}.npz"

# --- Stated measurement-noise assumption (readout circuit, NOT device mismatch) ---
# This models ADC quantization + comparator/thermal noise in the sensing circuit.
# 2% relative Gaussian noise on each current reading is a common, literature-typical
# assumption for a simple current-mode sense amplifier; flag this as an assumption,
# not a measured quantity, same as your Enhancement 3 process-variation caveat.
MEAS_NOISE_REL_STD = 0.02
R_REPEAT_READS = 1000   # virtual re-reads per chip, pure Python, no new ngspice calls

data = np.load(DATASET_FILE)
i_ref = data["i_ref"]        # shape (N_CHIPS, K_PAIRS)
i_target = data["i_target"]  # shape (N_CHIPS, K_PAIRS)
w_um = float(data["w_um"])
sigma_vth_mv = float(data["sigma_vth_mv"])
k_pairs = int(data["k_pairs"])
dvth_tattoo_mv = float(data["dvth_tattoo_mv"])

n_chips = i_ref.shape[0]
print(f"Design point: W={w_um}um, sigma_Vth={sigma_vth_mv:.2f}mV, K={k_pairs} pairs/chip, "
      f"N_CHIPS={n_chips}, tattoo target dvth={dvth_tattoo_mv}mV")
print(f"Measurement-noise assumption: {MEAS_NOISE_REL_STD*100:.1f}% relative Gaussian, "
      f"{R_REPEAT_READS} virtual re-reads/chip\n")

rng = np.random.default_rng(7)

# --- True chip signature (no measurement noise) -- this is the real, simulated physics ---
true_ratio_per_chip = (i_target / i_ref).mean(axis=1)  # average over K pairs, per chip

uniqueness_std = true_ratio_per_chip.std()
print(f"Uniqueness (inter-chip std of true tattoo ratio): {uniqueness_std:.4f}")
print(f"  (population mean = {true_ratio_per_chip.mean():.4f})")

# --- Reliability: R virtual noisy re-reads per chip, using the STATED noise model ---
per_chip_reliability_std = np.zeros(n_chips)
for chip in range(n_chips):
    i_ref_true = i_ref[chip]      # shape (K_PAIRS,)
    i_target_true = i_target[chip]

    reads = np.zeros(R_REPEAT_READS)
    for r in range(R_REPEAT_READS):
        noisy_ref = i_ref_true * (1 + rng.normal(0, MEAS_NOISE_REL_STD, size=K_PAIRS))
        noisy_target = i_target_true * (1 + rng.normal(0, MEAS_NOISE_REL_STD, size=K_PAIRS))
        reads[r] = (noisy_target / noisy_ref).mean()

    per_chip_reliability_std[chip] = reads.std()

reliability_std = per_chip_reliability_std.mean()
print(f"Reliability (mean intra-chip std across {R_REPEAT_READS} reads/chip): {reliability_std:.4f}")

d_prime = uniqueness_std / reliability_std
print(f"\nDiscriminability index d' = uniqueness / reliability = {d_prime:.3f}")

# Pairwise separation factor (independent cross-check, same style as Enhancement 3)
diffs = []
for i in range(n_chips):
    for j in range(i+1, n_chips):
        diffs.append(abs(true_ratio_per_chip[i] - true_ratio_per_chip[j]))
mean_pairwise_diff = np.mean(diffs)
separation_factor = mean_pairwise_diff / reliability_std
print(f"Pairwise separation factor (mean inter-chip diff / reliability std) = {separation_factor:.3f}")

print("\n--- Summary table ---")
print(f"{'Metric':45s}{'Value':>10s}")
print(f"{'Uniqueness (inter-chip std)':45s}{uniqueness_std:10.4f}")
print(f"{'Reliability (intra-chip std)':45s}{reliability_std:10.4f}")
print(f"{'Discriminability index d\'':45s}{d_prime:10.3f}")
print(f"{'Pairwise separation factor':45s}{separation_factor:10.3f}")