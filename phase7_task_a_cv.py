import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_pickle("unified_leakage_dataset.pkl")
task_a = df[df["application"] == "counterfeit_detection"].copy()

CATS = ["fresh", "light", "moderate", "severe"]
task_a["label_idx"] = task_a["ground_truth_label"].map({c: i for i, c in enumerate(CATS)})

X = task_a[["leakage_ratio"]].values
y = task_a["label_idx"].values
groups = task_a["chip_id"].values

N_FOLDS = 8  # 40 chips/category / 8 folds = 5 chips/category per fold, reasonable granularity
gkf = GroupKFold(n_splits=N_FOLDS)

def fit_thresholds(X, y):
    means = {i: X[y == i].mean() for i in range(4)}
    order = sorted(range(4), key=lambda i: means[i])
    thresholds = [(means[order[i]] + means[order[i+1]]) / 2 for i in range(3)]
    return order, thresholds

def classify_threshold(x, order, thresholds):
    for i, t in enumerate(thresholds):
        if x < t:
            return order[i]
    return order[-1]

model_fold_scores = {name: [] for name in
                      ["Threshold", "Logistic Regression", "Random Forest", "SVM (RBF)", "XGBoost"]}

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    del model_fold_scores["XGBoost"]
    print("xgboost not installed -- excluding from comparison\n")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    order, thresholds = fit_thresholds(X_train.ravel(), y_train)
    pred = np.array([classify_threshold(x, order, thresholds) for x in X_test.ravel()])
    model_fold_scores["Threshold"].append(accuracy_score(y_test, pred))

    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    model_fold_scores["Logistic Regression"].append(accuracy_score(y_test, logreg.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42).fit(X_train, y_train)
    model_fold_scores["Random Forest"].append(accuracy_score(y_test, rf.predict(X_test)))

    svm = SVC(kernel="rbf", C=1.0).fit(X_train, y_train)
    model_fold_scores["SVM (RBF)"].append(accuracy_score(y_test, svm.predict(X_test)))

    if has_xgb:
        xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=42,
                             eval_metric="mlogloss", verbosity=0).fit(X_train, y_train)
        model_fold_scores["XGBoost"].append(accuracy_score(y_test, xgb.predict(X_test)))

print(f"=== {N_FOLDS}-fold GroupKFold cross-validation (grouped by chip_id) ===\n")
print(f"{'Model':22s} {'Mean Acc':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
summary = {}
for name, scores in model_fold_scores.items():
    scores = np.array(scores)
    summary[name] = scores.mean()
    print(f"{name:22s} {scores.mean()*100:9.1f}% {scores.std()*100:7.1f}% "
          f"{scores.min()*100:7.1f}% {scores.max()*100:7.1f}%")

best = max(summary, key=summary.get)
threshold_mean = summary["Threshold"]
print(f"\nBest mean: {best} ({summary[best]*100:.1f}%)")
print(f"Threshold mean: {threshold_mean*100:.1f}%")
gap = summary[best] - threshold_mean
print(f"Gap: {gap*100:+.1f} points")

# Rough significance check: is the gap smaller than the fold-to-fold std of either model?
best_std = np.std(model_fold_scores[best])
thresh_std = np.std(model_fold_scores["Threshold"])
print(f"\nThreshold fold-to-fold std: {thresh_std*100:.1f} points")
print(f"{best} fold-to-fold std: {best_std*100:.1f} points")
if abs(gap) < max(best_std, thresh_std):
    print("\n-> Gap is smaller than the fold-to-fold variability of either model.")
    print("   Not a statistically meaningful difference -- the honest conclusion is that")
    print("   the threshold method and the best ML model perform equivalently on this")
    print("   single-feature task. Recommend reporting the threshold as the primary method")
    print("   (simpler, more interpretable, matches your 'don't force complexity' philosophy).")
else:
    print("\n-> Gap exceeds fold-to-fold variability -- a real, defensible improvement.")