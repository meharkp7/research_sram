import numpy as np

DATASET_FILE = "phase4_corrected_dataset_w3200.npz"  # change to w2000 file to compare
K_LIST = [1, 2, 4, 5, 10]

data = np.load(DATASET_FILE, allow_pickle=True)
ratio_all = data["ratio"]
label_all = data["label"]
w_um = float(data["w_um"])
sigma_vth_mv = float(data["sigma_vth_mv"])

CATS = ["fresh", "light", "moderate", "severe"]

def d_prime(a, b):
    return abs(np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

def build_grouped(K):
    """Group each category's samples into batches of K, average the ratio within each batch."""
    grouped = {}
    for cat in CATS:
        r = ratio_all[label_all == cat]
        n_groups = len(r) // K
        if n_groups < 3:
            return None  # not enough groups to be meaningful
        g = r[:n_groups*K].reshape(n_groups, K).mean(axis=1)
        grouped[cat] = g
    return grouped

def classify_and_score(grouped):
    means = {c: grouped[c].mean() for c in CATS}
    stds = {c: grouped[c].std() for c in CATS}
    order = sorted(CATS, key=lambda c: means[c])
    thresholds = [(means[a] + means[b]) / 2 for a, b in zip(order[:-1], order[1:])]

    def classify(r):
        for cat, t in zip(order[:-1], thresholds):
            if r < t:
                return cat
        return order[-1]

    all_ratio = np.concatenate([grouped[c] for c in CATS])
    all_label = np.concatenate([[c]*len(grouped[c]) for c in CATS])
    pred = np.array([classify(r) for r in all_ratio])

    correct = np.sum(pred == all_label)
    acc = correct / len(all_label)

    # Binary AUC: moderate+severe vs fresh+light
    suspect = np.isin(all_label, ["moderate", "severe"]).astype(int)
    score = all_ratio
    if score[suspect == 1].mean() > score[suspect == 0].mean():
        s = score
    else:
        s = -score
    thr_roc = np.sort(np.unique(s))
    P, N = suspect.sum(), len(suspect) - suspect.sum()
    tpr, fpr = [], []
    for t in thr_roc:
        pos = s >= t
        tpr.append(np.sum(pos & (suspect==1)) / P)
        fpr.append(np.sum(pos & (suspect==0)) / N)
    idx = np.argsort(fpr)
    fpr_s, tpr_s = np.array(fpr)[idx], np.array(tpr)[idx]
    auc = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(tpr_s, fpr_s)

    dp_fresh_severe = d_prime(grouped["fresh"], grouped["severe"])
    dp_adjacent = [d_prime(grouped[a], grouped[b]) for a, b in zip(CATS[:-1], CATS[1:])]

    return acc, auc, dp_fresh_severe, dp_adjacent, len(grouped["fresh"])

print(f"Dataset: {DATASET_FILE} (W={w_um}um, sigma_Vth={sigma_vth_mv:.2f}mV)\n")
print(f"{'K':>3} {'n/cat':>6} {'d(f-s)':>8} {'d(adjacent, min)':>18} {'4way acc':>10} {'AUC':>8}")
for K in K_LIST:
    grouped = build_grouped(K)
    if grouped is None:
        print(f"{K:3d}  -- not enough samples for meaningful grouping, skipping")
        continue
    acc, auc, dp_fs, dp_adj, n = classify_and_score(grouped)
    print(f"{K:3d} {n:6d} {dp_fs:8.3f} {min(dp_adj):18.3f} {acc*100:9.1f}% {auc:8.3f}")

print("\nNote: n/cat shrinks as K grows (only 100 independent draws per category available),")
print("so large-K rows are noisier estimates -- treat trend direction as the main signal,")
print("not the exact numbers at K=10.")