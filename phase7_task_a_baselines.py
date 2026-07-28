import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_pickle("unified_leakage_dataset.pkl")

task_a = df[df["application"] == "counterfeit_detection"].copy()
print(f"Task A samples: {len(task_a)}, unique chips: {task_a['chip_id'].nunique()}")

CATS = ["fresh", "light", "moderate", "severe"]
task_a["label_idx"] = task_a["ground_truth_label"].map({c: i for i, c in enumerate(CATS)})

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(task_a, groups=task_a["chip_id"]))
train_df, test_df = task_a.iloc[train_idx], task_a.iloc[test_idx]
print(f"Train: {len(train_df)} samples ({train_df['chip_id'].nunique()} chips), "
      f"Test: {len(test_df)} samples ({test_df['chip_id'].nunique()} chips)")
assert set(train_df["chip_id"]) & set(test_df["chip_id"]) == set(), "CHIP LEAKAGE DETECTED"
print("Confirmed: zero chip_id overlap between train and test\n")

X_train = train_df[["leakage_ratio"]].values
X_test = test_df[["leakage_ratio"]].values
y_train = train_df["label_idx"].values
y_test = test_df["label_idx"].values

results = {}

# --- 1. Grid-searched thresholds (your original methodology, fit on TRAIN only) ---
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

order, thresholds = fit_thresholds(X_train.ravel(), y_train)
pred = np.array([classify_threshold(x, order, thresholds) for x in X_test.ravel()])
results["Threshold"] = accuracy_score(y_test, pred)

# --- 2. Logistic Regression ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
results["Logistic Regression"] = accuracy_score(y_test, logreg.predict(X_test))

# --- 3. Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
rf.fit(X_train, y_train)
results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test))

# --- 4. SVM (RBF kernel) ---
svm = SVC(kernel="rbf", C=1.0)
svm.fit(X_train, y_train)
results["SVM (RBF)"] = accuracy_score(y_test, svm.predict(X_test))

# --- 5. XGBoost (if installed) ---
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=42,
                         use_label_encoder=False, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)
    results["XGBoost"] = accuracy_score(y_test, xgb.predict(X_test))
except ImportError:
    print("xgboost not installed (pip install xgboost) -- skipping\n")

print("=== Model comparison (Task A: 4-way aging classification, single feature) ===")
for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"  {name:22s}: {acc*100:5.1f}%")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name} ({results[best_name]*100:.1f}%)")
print(f"Threshold baseline:   {results['Threshold']*100:.1f}%")
print(f"Gap (best - threshold): {(results[best_name]-results['Threshold'])*100:+.1f} points")

if results[best_name] - results["Threshold"] < 0.03:
    print("\n-> Gap is small (<3 points): the threshold method already captures most of what's")
    print("   learnable from a single feature. This matches the expected 1-D-separability")
    print("   result and is a legitimate finding, not a failure to find a better model.")
else:
    print("\n-> Meaningful gap found -- worth investigating what the winning model is")
    print("   learning that a simple threshold misses (check its decision boundary).")

print("\n=== Best model's confusion matrix on held-out chips ===")
best_model_preds = {
    "Threshold": pred, "Logistic Regression": logreg.predict(X_test),
    "Random Forest": rf.predict(X_test), "SVM (RBF)": svm.predict(X_test),
}
if "XGBoost" in results:
    best_model_preds["XGBoost"] = xgb.predict(X_test)
print(confusion_matrix(y_test, best_model_preds[best_name]))
print(classification_report(y_test, best_model_preds[best_name], target_names=CATS, zero_division=0))