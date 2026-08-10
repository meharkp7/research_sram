import numpy as np
import pandas as pd
from scipy.stats import norm

N_BOOTSTRAP = 5000
rng = np.random.default_rng(2026)

df = pd.read_pickle("unified_leakage_dataset.pkl")

def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05):
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                            for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
    return boot_means.mean(), lo, hi

# ============ TASK A: accuracy + AUC, bootstrapped at the CHIP level ============
print("=" * 60)
print("TASK A: Counterfeit/aging classification")
print("=" * 60)

task_a = df[df["application"] == "counterfeit_detection"].copy()
CATS = ["fresh", "light", "moderate", "severe"]
task_a["label_idx"] = task_a["ground_truth_label"].map({c: i for i, c in enumerate(CATS)})
chip_ids = task_a["chip_id"].unique()

def fit_thresholds(X, y):
    means = {i: X[y == i].mean() for i in range(4)}
    order = sorted(range(4), key=lambda i: means[i])
    thr = [(means[order[i]] + means[order[i+1]]) / 2 for i in range(3)]
    return order, thr

def classify(x, order, thr):
    for i, t in enumerate(thr):
        if x < t:
            return order[i]
    return order[-1]

def compute_acc_and_auc(sample_df):
    X = sample_df["leakage_ratio"].values
    y = sample_df["label_idx"].values
    order, thr = fit_thresholds(X, y)
    pred = np.array([classify(x, order, thr) for x in X])
    acc = np.mean(pred == y)

    suspect = np.isin(y, [CATS.index("moderate"), CATS.index("severe")]).astype(int)
    score = X if X[suspect == 1].mean() > X[suspect == 0].mean() else -X
    thr_roc = np.sort(np.unique(score))
    P, N = suspect.sum(), len(suspect) - suspect.sum()
    if P == 0 or N == 0:
        return acc, np.nan
    tpr, fpr = [], []
    for t in thr_roc:
        pos = score >= t
        tpr.append(np.sum(pos & (suspect == 1)) / P)
        fpr.append(np.sum(pos & (suspect == 0)) / N)
    idx = np.argsort(fpr)
    auc = np.trapezoid(np.array(tpr)[idx], np.array(fpr)[idx]) if hasattr(np, "trapezoid") \
        else np.trapz(np.array(tpr)[idx], np.array(fpr)[idx])
    return acc, auc

acc_point, auc_point = compute_acc_and_auc(task_a)
print(f"Point estimates: accuracy={acc_point*100:.1f}%, AUC={auc_point:.3f}\n")

boot_accs, boot_aucs = [], []
for _ in range(N_BOOTSTRAP):
    sample_chips = rng.choice(chip_ids, size=len(chip_ids), replace=True)
    sample_df = pd.concat([task_a[task_a["chip_id"] == c] for c in sample_chips])
    acc, auc = compute_acc_and_auc(sample_df)
    boot_accs.append(acc)
    boot_aucs.append(auc)

boot_accs, boot_aucs = np.array(boot_accs), np.array(boot_aucs)
acc_lo, acc_hi = np.percentile(boot_accs, [2.5, 97.5])
auc_lo, auc_hi = np.percentile(boot_aucs[~np.isnan(boot_aucs)], [2.5, 97.5])
print(f"Bootstrap 95% CI, accuracy: [{acc_lo*100:.1f}%, {acc_hi*100:.1f}%]")
print(f"Bootstrap 95% CI, AUC:      [{auc_lo:.3f}, {auc_hi:.3f}]")

# ============ TASK B: accuracy, bootstrapped at the CHIP level ============
print("\n" + "=" * 60)
print("TASK B: Tattoo bit classification")
print("=" * 60)

task_b = df[df["application"] == "digital_tattoo_message"].copy()
chip_ids_b = task_b["chip_id"].unique()

def fit_threshold_binary(X, y):
    m1, m0 = X[y == 1].mean(), X[y == 0].mean()
    return (m1 + m0) / 2, m1 > m0

def compute_acc_b(sample_df):
    X = sample_df["leakage_ratio"].values
    y = sample_df["tattoo_bit"].values.astype(int)
    thr, higher_is_1 = fit_threshold_binary(X, y)
    pred = (X >= thr).astype(int) if higher_is_1 else (X <= thr).astype(int)
    return np.mean(pred == y)

acc_b_point = compute_acc_b(task_b)
print(f"Point estimate accuracy: {acc_b_point*100:.2f}%\n")

boot_accs_b = []
for _ in range(N_BOOTSTRAP):
    sample_chips = rng.choice(chip_ids_b, size=len(chip_ids_b), replace=True)
    sample_df = pd.concat([task_b[task_b["chip_id"] == c] for c in sample_chips])
    boot_accs_b.append(compute_acc_b(sample_df))

boot_accs_b = np.array(boot_accs_b)
acc_b_lo, acc_b_hi = np.percentile(boot_accs_b, [2.5, 97.5])
print(f"Bootstrap 95% CI, accuracy: [{acc_b_lo*100:.2f}%, {acc_b_hi*100:.2f}%]")

# ============ AUTHENTICATION FAR/FRR: bootstrap from the raw 100-chip uniqueness data ============
print("\n" + "=" * 60)
print("Authentication FAR/FRR (Segment 5.5, k=3, N=8)")
print("=" * 60)

try:
    d = np.load("phase5_tattoo_uniqueness_K4_w3200.npz")
    i_ref, i_target = d["i_ref"], d["i_target"]  # shape (n_chips, K_PAIRS) -- raw currents, not pre-averaged
    n_chips_auth = i_ref.shape[0]
    K_PAIRS = i_ref.shape[1]
    true_ratio_per_chip = (i_target / i_ref).mean(axis=1)

    K_SITES, K_SIGMA = 8, 3
    MEAS_NOISE = 0.02
    R_READS = 200  # reduced from Segment 5.3's 1000 for bootstrap speed, still ample for a stable std estimate

    def compute_far_frr(idx_sample):
        """idx_sample: indices into the ORIGINAL i_ref/i_target arrays (supports resampling with replacement)"""
        sample_i_ref = i_ref[idx_sample]      # shape (n, K_PAIRS)
        sample_i_target = i_target[idx_sample]
        chip_ratios = (sample_i_target / sample_i_ref).mean(axis=1)
        uniqueness_std = chip_ratios.std()

        # Reliability: properly simulate K_PAIRS independent noisy reads per chip, THEN average --
        # matching Segment 5.3's original methodology exactly, not adding noise to the pre-averaged ratio.
        per_chip_std = np.zeros(len(idx_sample))
        for i in range(len(idx_sample)):
            i_ref_true = sample_i_ref[i]       # shape (K_PAIRS,)
            i_target_true = sample_i_target[i]
            reads = np.zeros(R_READS)
            for r in range(R_READS):
                noisy_ref = i_ref_true * (1 + rng.normal(0, MEAS_NOISE, size=K_PAIRS))
                noisy_target = i_target_true * (1 + rng.normal(0, MEAS_NOISE, size=K_PAIRS))
                reads[r] = (noisy_target / noisy_ref).mean()
            per_chip_std[i] = reads.std()
        reliability_std = per_chip_std.mean()

        sigma_genuine = np.sqrt(2) * reliability_std
        sigma_impostor = np.sqrt(2 * uniqueness_std**2 + 2 * reliability_std**2)
        threshold = K_SIGMA * sigma_genuine
        frr_1 = 2 * (1 - norm.cdf(threshold / sigma_genuine))
        far_1 = 1 - 2 * (1 - norm.cdf(threshold / sigma_impostor))
        far_n = far_1 ** K_SITES
        frr_n = 1 - (1 - frr_1) ** K_SITES
        return far_n, frr_n

    far_point, frr_point = compute_far_frr(np.arange(n_chips_auth))
    print(f"Point estimates: FAR_8={far_point:.3e}, FRR_8={frr_point*100:.2f}%")
    print(f"(Compare to Segment 5.5's original: FAR_8=6.58e-07, FRR_8=2.14% -- should match closely now)\n")

    boot_far, boot_frr = [], []
    for _ in range(300):  # fewer iterations -- this inner loop (R_READS x n_chips) is the expensive part
        sample_idx = rng.integers(0, n_chips_auth, size=n_chips_auth)
        far, frr = compute_far_frr(sample_idx)
        boot_far.append(far)
        boot_frr.append(frr)

    boot_far, boot_frr = np.array(boot_far), np.array(boot_frr)
    far_lo, far_hi = np.percentile(boot_far, [2.5, 97.5])
    frr_lo, frr_hi = np.percentile(boot_frr, [2.5, 97.5])
    print(f"Bootstrap 95% CI, FAR_8: [{far_lo:.3e}, {far_hi:.3e}]")
    print(f"Bootstrap 95% CI, FRR_8: [{frr_lo*100:.2f}%, {frr_hi*100:.2f}%]")
    print(f"\nNote: FRR_8 varies little across bootstrap samples by construction -- the matching")
    print(f"threshold is DEFINED as k_sigma x sigma_genuine, so threshold/sigma_genuine is fixed")
    print(f"at exactly k_sigma=3 regardless of the data, making FRR_1 (and thus FRR_8) essentially")
    print(f"data-independent given this threshold-setting convention. Only FAR genuinely reflects")
    print(f"the resampled uniqueness/reliability values.")
except FileNotFoundError:
    print("phase5_tattoo_uniqueness_K4_w3200.npz not found -- skipping FAR/FRR bootstrap")

print("\n=== Summary table for the manuscript ===")
print(f"Task A accuracy:  {acc_point*100:.1f}% [{acc_lo*100:.1f}%, {acc_hi*100:.1f}%]")
print(f"Task A AUC:        {auc_point:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")
print(f"Task B accuracy:  {acc_b_point*100:.2f}% [{acc_b_lo*100:.2f}%, {acc_b_hi*100:.2f}%]")