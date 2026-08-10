import numpy as np

d = np.load("phase8_asymmetric_correlation.npz")
chip_idx, bit_val, ratio_arr = d["chip_idx"], d["bit_value"], d["ratio"]
n_chips, bits_per_chip = int(d["n_chips"]), int(d["bits_per_chip"])

pop_mean_0 = ratio_arr[bit_val == 0].mean()
pop_mean_1 = ratio_arr[bit_val == 1].mean()
naive_threshold = (pop_mean_0 + pop_mean_1) / 2
naive_higher_is_1 = pop_mean_1 > pop_mean_0
expected = np.where(bit_val == 1, pop_mean_1, pop_mean_0)
residual = ratio_arr - expected

rng = np.random.default_rng(7)
M_VALUES = [5, 10, 15, 20, 25]  # up to bits_per_chip - a few held out for testing

print(f"{'M observed':>12} {'held out':>10} {'naive acc':>12} {'attack acc':>12} {'gap':>8}")
for M in M_VALUES:
    if M >= bits_per_chip - 2:
        continue
    naive_accs, attack_accs = [], []
    for chip in range(n_chips):
        mask = chip_idx == chip
        chip_ratio = ratio_arr[mask]
        chip_bitval = bit_val[mask]
        chip_residual = residual[mask]
        idx = rng.permutation(len(chip_ratio))
        obs_idx, held_idx = idx[:M], idx[M:]

        pred_naive = (chip_ratio[held_idx] >= naive_threshold).astype(int) if naive_higher_is_1 \
                     else (chip_ratio[held_idx] <= naive_threshold).astype(int)
        naive_accs.append(np.mean(pred_naive == chip_bitval[held_idx]))

        est_bias = chip_residual[obs_idx].mean()
        adj_threshold = naive_threshold + est_bias
        pred_attack = (chip_ratio[held_idx] >= adj_threshold).astype(int) if naive_higher_is_1 \
                      else (chip_ratio[held_idx] <= adj_threshold).astype(int)
        attack_accs.append(np.mean(pred_attack == chip_bitval[held_idx]))

    naive_accs, attack_accs = np.array(naive_accs), np.array(attack_accs)
    gap = attack_accs.mean() - naive_accs.mean()
    print(f"{M:12d} {bits_per_chip-M:10d} {naive_accs.mean()*100:11.2f}% "
          f"{attack_accs.mean()*100:11.2f}% {gap*100:+7.2f}")

print("\nIf the gap grows with M, the leakage is real and exploitable given enough observed")
print("cells -- meaning good layout matching is a genuine security requirement, not just a")
print("statistical curiosity. If the gap stays near zero regardless of M, the 60mV signal-to-")
print("noise ratio is simply too favorable for this class of leakage to matter practically,")
print("even though it's statistically detectable in aggregate.")