import numpy as np
import pandas as pd

d = np.load("phase8_correlated_mismatch.npz")
chip_idx, bit_pos, bit_val, ratio = d["chip_idx"], d["bit_position"], d["bit_value"], d["ratio"]
n_chips, bits_per_chip = int(d["n_chips"]), int(d["bits_per_chip"])
sigma_local, sigma_systematic = float(d["sigma_local_mv"]), float(d["sigma_systematic_mv"])

df = pd.DataFrame({"chip": chip_idx, "bit_pos": bit_pos, "bit_val": bit_val, "ratio": ratio})

print(f"Loaded {len(df)} samples, {n_chips} chips, {bits_per_chip} bits/chip")
print(f"sigma_local={sigma_local:.2f}mV, sigma_systematic={sigma_systematic:.2f}mV\n")

# ============ STEP 1: Does the systematic offset leak into the ratio at all? ============
# For each chip, compute the mean RESIDUAL (deviation from the population-level bit=0/bit=1
# means) across that chip's cells. If the systematic offset is fully cancelled by the ratio,
# these per-chip mean residuals should be close to zero and uncorrelated with anything chip-specific
# (just reflect ordinary sampling noise). If it leaks, chips will show a systematic per-chip bias.

pop_mean_0 = df[df.bit_val == 0]["ratio"].mean()
pop_mean_1 = df[df.bit_val == 1]["ratio"].mean()
print(f"Population means: bit=0: {pop_mean_0:.4f}, bit=1: {pop_mean_1:.4f}\n")

df["expected"] = np.where(df.bit_val == 1, pop_mean_1, pop_mean_0)
df["residual"] = df["ratio"] - df["expected"]

chip_mean_residual = df.groupby("chip")["residual"].mean()
print(f"Per-chip mean residual (should be ~0 with small scatter if fully cancelled):")
print(f"  mean: {chip_mean_residual.mean():.5f}")
print(f"  std across chips: {chip_mean_residual.std():.5f}")
print(f"  min: {chip_mean_residual.min():.5f}, max: {chip_mean_residual.max():.5f}")

# Expected scatter if there were NO systematic leakage at all: just sampling noise of the mean
# of `bits_per_chip` independent residuals, i.e. (population residual std) / sqrt(bits_per_chip)
population_residual_std = df["residual"].std()
expected_noise_only_std = population_residual_std / np.sqrt(bits_per_chip)
print(f"\n  Expected std IF fully cancelled (pure sampling noise): {expected_noise_only_std:.5f}")

leakage_ratio_metric = chip_mean_residual.std() / expected_noise_only_std
print(f"  Observed / expected-if-cancelled ratio: {leakage_ratio_metric:.2f}x")
if leakage_ratio_metric < 1.5:
    print("  -> Close to 1x: consistent with FULL cancellation, no meaningful systematic leakage.")
    print("     This is a GOOD finding: differential ratio sensing appears robust to shared")
    print("     die-level mismatch, not just environmental variation.")
else:
    print("  -> Meaningfully above 1x: systematic offset is LEAKING into the ratio despite")
    print("     cancellation, i.e. the shared-offset assumption doesn't fully cancel in this")
    print("     real BSIM model (likely due to second-order/non-linear effects). Proceeding")
    print("     to test whether this is exploitable.")

# ============ STEP 2: Modeling attack -- can observing M cells predict a held-out cell? ============
# For each chip, use M observed cells' residuals to estimate that chip's systematic bias,
# then check whether this improves prediction of a held-out cell's bit value versus the
# naive population-threshold baseline.

M_OBSERVED = 20  # attacker observes 20 of the 64 cells (with known ground truth, e.g. via
                  # partial extraction or side channel) and tries to improve prediction on the rest

naive_threshold = (pop_mean_0 + pop_mean_1) / 2
naive_higher_is_1 = pop_mean_1 > pop_mean_0

results_naive, results_attack = [], []

rng = np.random.default_rng(42)
for chip in range(n_chips):
    chip_df = df[df.chip == chip].reset_index(drop=True)
    idx = rng.permutation(len(chip_df))
    observed_idx, held_out_idx = idx[:M_OBSERVED], idx[M_OBSERVED:]

    observed = chip_df.iloc[observed_idx]
    held_out = chip_df.iloc[held_out_idx]

    # Naive baseline: population threshold, ignoring any chip-specific info
    naive_pred = (held_out["ratio"] >= naive_threshold).astype(int) if naive_higher_is_1 \
                 else (held_out["ratio"] <= naive_threshold).astype(int)
    naive_acc = (naive_pred.values == held_out["bit_val"].values).mean()
    results_naive.append(naive_acc)

    # Attack: estimate this chip's systematic bias from observed cells' residuals,
    # shift the threshold by that estimated bias, then predict held-out cells
    estimated_bias = observed["residual"].mean()
    adjusted_threshold = naive_threshold + estimated_bias
    attack_pred = (held_out["ratio"] >= adjusted_threshold).astype(int) if naive_higher_is_1 \
                  else (held_out["ratio"] <= adjusted_threshold).astype(int)
    attack_acc = (attack_pred.values == held_out["bit_val"].values).mean()
    results_attack.append(attack_acc)

naive_acc_arr = np.array(results_naive)
attack_acc_arr = np.array(results_attack)

print(f"\n=== Modeling attack test ({M_OBSERVED} observed cells, {bits_per_chip - M_OBSERVED} held out) ===")
print(f"Naive (population threshold) accuracy:  {naive_acc_arr.mean()*100:.2f}% "
      f"(std across chips: {naive_acc_arr.std()*100:.2f}%)")
print(f"Attack (bias-adjusted threshold) accuracy: {attack_acc_arr.mean()*100:.2f}% "
      f"(std across chips: {attack_acc_arr.std()*100:.2f}%)")
gap = attack_acc_arr.mean() - naive_acc_arr.mean()
print(f"Gap: {gap*100:+.2f} points")

if abs(gap) < max(naive_acc_arr.std(), attack_acc_arr.std()):
    print("\n-> Gap is within fold-to-fold noise. The systematic-offset modeling attack provides")
    print("   no meaningful advantage over the naive baseline -- consistent with the ratio metric")
    print("   cancelling shared per-chip mismatch, same as it does for temperature/voltage.")
else:
    print("\n-> Meaningful gap: exploiting the systematic offset genuinely improves an attacker's")
    print("   prediction accuracy. This is an important, real finding for the security section --")
    print("   it means Segment 8.3's FAR_N=FAR_1^N formula (assumed site independence) is optimistic,")
    print("   and correlated mismatch should be accounted for in the security claim.")