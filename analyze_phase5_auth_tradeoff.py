import numpy as np
from scipy.stats import norm

K_PAIRS = 4
W_NM = 3200
DATASET_FILE = f"phase5_tattoo_uniqueness_K{K_PAIRS}_w{W_NM}.npz"
MEAS_NOISE_REL_STD = 0.02  # matches the design-point noise level used in Segment 5.3/5.4

data = np.load(DATASET_FILE)
i_ref = data["i_ref"]
i_target = data["i_target"]
n_chips = i_ref.shape[0]

true_ratio_per_chip = (i_target / i_ref).mean(axis=1)
uniqueness_std = true_ratio_per_chip.std()

# Reliability at the stated noise level (reuse the same estimation approach as Segment 5.3)
rng = np.random.default_rng(7)
R_READS = 500
per_chip_std = np.zeros(n_chips)
for chip in range(n_chips):
    reads = np.zeros(R_READS)
    for r in range(R_READS):
        noisy_ref = i_ref[chip] * (1 + rng.normal(0, MEAS_NOISE_REL_STD, size=K_PAIRS))
        noisy_target = i_target[chip] * (1 + rng.normal(0, MEAS_NOISE_REL_STD, size=K_PAIRS))
        reads[r] = (noisy_target / noisy_ref).mean()
    per_chip_std[chip] = reads.std()
reliability_std = per_chip_std.mean()

print(f"Uniqueness std: {uniqueness_std:.4f}, Reliability std: {reliability_std:.4f}\n")

# Per-site diff distributions:
#  Genuine (same chip, template vs verification read): diff ~ N(0, 2*reliability_std^2)
#  Impostor (different chip):                          diff ~ N(0, 2*uniqueness_std^2 + 2*reliability_std^2)
sigma_genuine = np.sqrt(2) * reliability_std
sigma_impostor = np.sqrt(2 * uniqueness_std**2 + 2 * reliability_std**2)

print(f"{'N sites':>8} {'k (sigma)':>10} {'FRR (1 site)':>14} {'FAR (1 site)':>14} "
      f"{'FRR_N (AND)':>14} {'FAR_N (AND)':>16}")

for k in [2, 3, 4]:
    for N in [1, 2, 4, 8, 16]:
        threshold = k * sigma_genuine  # matching window defined in genuine-noise sigmas
        frr_1 = 2 * (1 - norm.cdf(threshold / sigma_genuine))   # P(|diff| > threshold | genuine)
        far_1 = 2 * norm.cdf(threshold / sigma_impostor) - 1    # P(|diff| <= threshold | impostor)
        far_1 = 1 - far_1  # P(|diff| <= threshold) directly:
        far_1 = 2 * norm.cdf(-threshold / sigma_impostor)  # correct: P(|diff|<=thr) = 1 - 2*P(diff>thr) but simpler:
        far_1 = 1 - 2 * (1 - norm.cdf(threshold / sigma_impostor))

        frr_n = 1 - (1 - frr_1) ** N
        far_n = far_1 ** N
        print(f"{N:8d} {k:10d} {frr_1*100:13.2f}% {far_1*100:13.2f}% "
              f"{frr_n*100:13.2f}% {far_n:16.2e}")
    print()

print("Recommendation: pick the smallest N/k combination where FAR_N is acceptably")
print("low for your threat model (e.g. <1e-6) while FRR_N stays low enough to be")
print("usable (a few % is normal for PUF-style authentication with retry allowed).")