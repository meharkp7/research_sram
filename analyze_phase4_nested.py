import numpy as np

K_PAIRS = 4
W_NM = 3200
DATASET_FILE = f"phase4_nested_K{K_PAIRS}_w{W_NM}.npz"

data = np.load(DATASET_FILE, allow_pickle=True)
ratio = data["chip_ratio"]
label = data["label"]
w_um = float(data["w_um"])
sigma_vth_mv = float(data["sigma_vth_mv"])
k_pairs = int(data["k_pairs"])

CATS = ["fresh", "light", "moderate", "severe"]
print(f"Design point: W={w_um}um, sigma_Vth={sigma_vth_mv:.2f}mV, K={k_pairs} pairs/chip\n")

means, stds = {}, {}
for cat in CATS:
    m = ratio[label == cat]
    means[cat] = m.mean()
    stds[cat] = m.std()
    print(f"{cat:10s}: n={len(m):3d}, mean={means[cat]:.4f}, std={stds[cat]:.4f}, "
          f"CV={100*stds[cat]/means[cat]:.1f}%")

def d_prime(a_mean, a_std, b_mean, b_std):
    return abs(a_mean - b_mean) / np.sqrt((a_std**2 + b_std**2) / 2)

print("\nAdjacent-category d':")
for a, b in zip(CATS[:-1], CATS[1:]):
    dp = d_prime(means[a], stds[a], means[b], stds[b])
    print(f"  {a} vs {b}: d' = {dp:.3f}")
print(f"  fresh vs severe: d' = {d_prime(means['fresh'],stds['fresh'],means['severe'],stds['severe']):.3f}")

order = sorted(CATS, key=lambda c: means[c])
thresholds = [(means[a] + means[b]) / 2 for a, b in zip(order[:-1], order[1:])]
print(f"\nCategory order (ascending ratio): {order}")
print(f"Thresholds: {[f'{t:.4f}' for t in thresholds]}")

def classify(r):
    for cat, t in zip(order[:-1], thresholds):
        if r < t:
            return cat
    return order[-1]

pred = np.array([classify(r) for r in ratio])
n = len(CATS)
cm = np.zeros((n, n), dtype=int)
idx = {c: i for i, c in enumerate(CATS)}
for t, p in zip(label, pred):
    cm[idx[t], idx[p]] += 1

print("\nConfusion matrix (rows=true, cols=predicted), order:", CATS)
print(cm)
acc = np.trace(cm) / cm.sum()
print(f"\nOverall accuracy: {acc*100:.1f}%")

print("\nPer-category precision/recall/F1:")
for i, cat in enumerate(CATS):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"  {cat:10s}: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

suspect = np.isin(label, ["moderate", "severe"]).astype(int)
score = ratio if ratio[suspect==1].mean() > ratio[suspect==0].mean() else -ratio
thr = np.sort(np.unique(score))
P, N = suspect.sum(), len(suspect) - suspect.sum()
tpr, fpr = [], []
for t in thr:
    pos = score >= t
    tpr.append(np.sum(pos & (suspect==1)) / P)
    fpr.append(np.sum(pos & (suspect==0)) / N)
order_idx = np.argsort(fpr)
fpr_s, tpr_s = np.array(fpr)[order_idx], np.array(tpr)[order_idx]
auc = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(tpr_s, fpr_s)
print(f"\nBinary counterfeit detection (moderate+severe vs fresh+light): AUC = {auc:.3f}")

print("\nOperating points:")
for target_fpr in [0.001, 0.01, 0.05, 0.10]:
    i2 = min(np.searchsorted(fpr_s, target_fpr), len(fpr_s)-1)
    print(f"  At FPR<={target_fpr*100:.1f}%: TPR = {tpr_s[i2]*100:.1f}%")