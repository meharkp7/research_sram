import numpy as np

fresh = np.load("mc_mismatch_results.npz")["i_refs"]           # same in all 3 files
light = np.load("mc_mismatch_results_light.npz")["i_targets"]
moderate = np.load("mc_mismatch_results_moderate.npz")["i_targets"]
severe = np.load("mc_mismatch_results.npz")["i_targets"]

categories = {"Fresh": fresh, "Light": light, "Moderate": moderate, "Severe": severe}

def d_prime(a, b):
    return abs(np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

print(f"{'Category':<10} {'Mean (nA)':<12} {'Std (nA)':<12}")
for name, arr in categories.items():
    print(f"{name:<10} {np.mean(arr)*1e9:<12.4f} {np.std(arr)*1e9:<12.4f}")

print("\n=== Pairwise discriminability (d') ===")
names = list(categories.keys())
for i in range(len(names) - 1):
    a, b = categories[names[i]], categories[names[i+1]]
    dp = d_prime(a, b)
    verdict = "separable" if dp > 2 else ("marginal" if dp > 1 else "NOT separable")
    print(f"{names[i]} vs {names[i+1]:<10}: d' = {dp:.3f}  ({verdict})")

print(f"\n{'Fresh vs Severe (extremes)':<28}: d' = {d_prime(fresh, severe):.3f}")
