import numpy as np

data = np.load("mc_mismatch_pelgrom_800nm.npz")
i_refs, i_targets = data["i_refs"], data["i_targets"]
ratio = i_targets / i_refs

# NOTE: this file only has ONE aging condition (severely aged target vs fresh ref).
# To compute d', you need a second dataset from the SAME W but a "fresh vs fresh"
# or a lighter-aging condition. If you don't have one yet, rerun mc_mismatch.py
# with VTH0_TARGET_NOMINAL = VTH0_REF_NOMINAL (both fresh) at W=800nm and save as
# mc_mismatch_pelgrom_800nm_bothfresh.npz -- that gives the "null" distribution
# to compare against.

def d_prime(a, b):
    return abs(np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

K_LIST = [1, 2, 3, 4, 5, 10, 20]
for K in K_LIST:
    n_groups = len(ratio) // K
    grouped = ratio[:n_groups*K].reshape(n_groups, K).mean(axis=1)
    print(f"K={K:2d}: n_groups={n_groups:3d}, mean={grouped.mean():.4f}, std={grouped.std():.4f}, "
          f"CV={100*grouped.std()/grouped.mean():.2f}%")