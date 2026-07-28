import numpy as np

data = np.load("phase4_corrected_dataset_w3200.npz", allow_pickle=True)
ratio = data["ratio"]
label = data["label"]
w_um = float(data["w_um"])
sigma_vth_mv = float(data["sigma_vth_mv"])

CATS = ["fresh", "light", "moderate", "severe"]

print(f"Design point: W={w_um}um, sigma_Vth={sigma_vth_mv:.2f}mV\n")

# --- Category stats ---
means, stds = {}, {}
for cat in CATS:
    m = ratio[label == cat]
    means[cat] = m.mean()
    stds[cat] = m.std()
    print(f"{cat:10s}: n={len(m):3d}, mean={means[cat]:.4f}, std={stds[cat]:.4f}, "
          f"CV={100*stds[cat]/means[cat]:.1f}%")

# --- d' between adjacent categories (ordered fresh -> severe) ---
print("\nAdjacent-category d':")
for a, b in zip(CATS[:-1], CATS[1:]):
    dprime = abs(means[a] - means[b]) / np.sqrt((stds[a]**2 + stds[b]**2) / 2)
    print(f"  {a} vs {b}: d' = {dprime:.3f}")
print(f"  fresh vs severe (headline uniqueness number): d' = "
      f"{abs(means['fresh']-means['severe'])/np.sqrt((stds['fresh']**2+stds['severe']**2)/2):.3f}")

# --- Grid-searched, variance-aware thresholds (same method as your original Phase 4) ---
# Sort category means descending (ratio drops as aging increases in this framework? check sign)
order = sorted(CATS, key=lambda c: means[c])  # ascending mean order
print(f"\nCategory order by ratio mean (ascending): {order}")

def midpoint_thresholds(order):
    thr = []
    for a, b in zip(order[:-1], order[1:]):
        thr.append((means[a] + means[b]) / 2)
    return thr

thresholds = midpoint_thresholds(order)
print(f"Midpoint thresholds: {[f'{t:.4f}' for t in thresholds]}")

def classify(r, order, thresholds):
    for cat, t in zip(order[:-1], thresholds):
        if r < t:
            return cat
    return order[-1]

pred = np.array([classify(r, order, thresholds) for r in ratio])

# --- Confusion matrix ---
n = len(CATS)
cm = np.zeros((n, n), dtype=int)
cat_idx = {c: i for i, c in enumerate(CATS)}
for true_c, pred_c in zip(label, pred):
    cm[cat_idx[true_c], cat_idx[pred_c]] += 1

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

# --- Binary counterfeit-detection ROC/AUC: moderate+severe vs fresh+light ---
suspect = np.isin(label, ["moderate", "severe"]).astype(int)  # 1 = suspicious/aged
score = ratio  # use raw ratio as the score; direction determined below

# Determine orientation: does suspicious class have higher or lower mean ratio?
if ratio[suspect == 1].mean() > ratio[suspect == 0].mean():
    score_for_roc = ratio
else:
    score_for_roc = -ratio

thresholds_roc = np.sort(np.unique(score_for_roc))
tpr_list, fpr_list = [], []
P = suspect.sum()
N = len(suspect) - P
for t in thresholds_roc:
    pred_pos = score_for_roc >= t
    tp = np.sum(pred_pos & (suspect == 1))
    fp = np.sum(pred_pos & (suspect == 0))
    tpr_list.append(tp / P)
    fpr_list.append(fp / N)

# Sort by fpr for AUC (trapezoidal)
order_idx = np.argsort(fpr_list)
fpr_sorted = np.array(fpr_list)[order_idx]
tpr_sorted = np.array(tpr_list)[order_idx]
auc = np.trapezoid(tpr_sorted, fpr_sorted) if hasattr(np, "trapezoid") else np.trapz(tpr_sorted, fpr_sorted)

print(f"\nBinary counterfeit detection (moderate+severe vs fresh+light): AUC = {auc:.3f}")

# Neyman-Pearson style operating points
print("\nOperating points:")
for target_fpr in [0.001, 0.01, 0.05, 0.10]:
    # find smallest fpr >= target, report corresponding tpr
    idx = np.searchsorted(fpr_sorted, target_fpr)
    idx = min(idx, len(fpr_sorted) - 1)
    print(f"  At FPR<={target_fpr*100:.1f}%: TPR = {tpr_sorted[idx]*100:.1f}%")

np.savez("phase4_corrected_results_w3200.npz",
         cm=cm, acc=acc, auc=auc, thresholds=thresholds, order=order)
print("\nSaved analysis results to phase4_corrected_results.npz")