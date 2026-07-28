import numpy as np

data = np.load("mc_mismatch_results.npz")
i_refs = data["i_refs"]       # fresh cell population (with mismatch)
i_targets = data["i_targets"] # severely aged cell population (with mismatch)

mean_fresh, std_fresh = np.mean(i_refs), np.std(i_refs)
mean_aged, std_aged = np.mean(i_targets), np.std(i_targets)

d_prime = abs(mean_aged - mean_fresh) / np.sqrt((std_fresh**2 + std_aged**2) / 2)

print(f"Fresh population : mean={mean_fresh*1e9:.4f} nA, std={std_fresh*1e9:.4f} nA")
print(f"Aged population  : mean={mean_aged*1e9:.4f} nA, std={std_aged*1e9:.4f} nA")
print(f"\nDiscriminability d' = {d_prime:.3f}")

if d_prime > 2:
    print("-> Comfortably separable from a single measurement.")
elif d_prime > 1:
    print("-> Marginally separable; averaging a few measurements would help.")
else:
    print("-> Substantial overlap; a single measurement is NOT reliable. Averaging needed.")

# How many repeated measurements N would it take to reach d'=2 (comfortable separation)?
# Averaging N independent measurements shrinks std by sqrt(N), so d' scales by sqrt(N).
if d_prime > 0 and d_prime < 2:
    n_needed = np.ceil((2 / d_prime) ** 2)
    print(f"\nMeasurements needed to average together for d'>=2: N = {int(n_needed)}")
