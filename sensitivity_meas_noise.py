import numpy as np

K_PAIRS = 4
W_NM = 3200
DATASET_FILE = f"phase5_tattoo_uniqueness_K{K_PAIRS}_w{W_NM}.npz"
R_REPEAT_READS = 500  # slightly reduced for speed across the sweep

NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]  # 0.5% to 20% relative Gaussian

data = np.load(DATASET_FILE)
i_ref = data["i_ref"]
i_target = data["i_target"]
n_chips = i_ref.shape[0]

true_ratio_per_chip = (i_target / i_ref).mean(axis=1)
uniqueness_std = true_ratio_per_chip.std()
print(f"Uniqueness (fixed, from real mismatch physics): {uniqueness_std:.4f}\n")

rng = np.random.default_rng(7)

print(f"{'Noise level':>12} {'Reliability std':>16} {'d prime':>10}")
for noise in NOISE_LEVELS:
    per_chip_std = np.zeros(n_chips)
    for chip in range(n_chips):
        i_ref_true = i_ref[chip]
        i_target_true = i_target[chip]
        reads = np.zeros(R_REPEAT_READS)
        for r in range(R_REPEAT_READS):
            noisy_ref = i_ref_true * (1 + rng.normal(0, noise, size=K_PAIRS))
            noisy_target = i_target_true * (1 + rng.normal(0, noise, size=K_PAIRS))
            reads[r] = (noisy_target / noisy_ref).mean()
        per_chip_std[chip] = reads.std()

    reliability_std = per_chip_std.mean()
    dprime = uniqueness_std / reliability_std
    print(f"{noise*100:11.1f}% {reliability_std:16.4f} {dprime:10.3f}")

print("\nIf d' stays well above ~3-5 even at the higher noise levels, the uniqueness")
print("claim is robust to the instrumentation-noise assumption. If it collapses")
print("quickly, the headline number is fragile and the assumption needs a real")
print("citation (e.g. a specific sense-amp/ADC noise spec) rather than a placeholder.")