import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

df = pd.read_pickle("unified_leakage_dataset.pkl")
task_b = df[df["application"] == "digital_tattoo_message"].copy()

X = task_b[["leakage_ratio"]].values
y = task_b["tattoo_bit"].values.astype(int)
groups = task_b["chip_id"].values

def fit_threshold(X, y):
    mean1, mean0 = X[y == 1].mean(), X[y == 0].mean()
    thr = (mean1 + mean0) / 2
    return thr, mean1 > mean0

def classify_threshold(x, thr, higher_is_1):
    return int(x >= thr) if higher_is_1 else int(x <= thr)

logo = LeaveOneGroupOut()
print(f"{'Held-out chip':30s} {'n samples':>10} {'accuracy':>10} {'errors':>8}")

per_chip_results = []
for train_idx, test_idx in logo.split(X, y, groups):
    test_chip = groups[test_idx][0]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    thr, higher_is_1 = fit_threshold(X_train.ravel(), y_train)
    pred = np.array([classify_threshold(x, thr, higher_is_1) for x in X_test.ravel()])
    acc = accuracy_score(y_test, pred)
    n_errors = np.sum(pred != y_test)
    per_chip_results.append((test_chip, len(test_idx), acc, n_errors))
    print(f"{test_chip:30s} {len(test_idx):10d} {acc*100:9.2f}% {n_errors:8d}")

accs = np.array([r[2] for r in per_chip_results])
print(f"\nAcross {len(per_chip_results)} held-out chips (leave-one-chip-out):")
print(f"Mean accuracy: {accs.mean()*100:.2f}%, std: {accs.std()*100:.2f}%, "
      f"min: {accs.min()*100:.2f}%, max: {accs.max()*100:.2f}%")
print(f"\nIf every chip individually lands in a similar, consistently-high range (not just")
print(f"the average), that confirms this is a real, chip-independent property of the design")
print(f"-- not one easy chip masking a problem elsewhere.")