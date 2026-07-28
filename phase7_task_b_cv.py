import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_pickle("unified_leakage_dataset.pkl")
task_b = df[df["application"] == "digital_tattoo_message"].copy()

print(f"Task B samples: {len(task_b)}, unique chips: {task_b['chip_id'].nunique()}")
print(task_b.groupby("chip_id").size().sort_values(ascending=False))

X = task_b[["leakage_ratio"]].values
y = task_b["tattoo_bit"].values.astype(int)
groups = task_b["chip_id"].values

N_FOLDS = 6  # 18 chips / 6 folds = 3 chips held out per fold
gkf = GroupKFold(n_splits=N_FOLDS)

def fit_threshold(X, y):
    mean1, mean0 = X[y == 1].mean(), X[y == 0].mean()
    thr = (mean1 + mean0) / 2
    return thr, mean1 > mean0

def classify_threshold(x, thr, higher_is_1):
    if higher_is_1:
        return int(x >= thr)
    return int(x <= thr)

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("xgboost not installed -- excluding\n")

model_names = ["Threshold", "Logistic Regression", "Random Forest", "SVM (RBF)", "DNN (small MLP)"]
if has_xgb:
    model_names.append("XGBoost")
model_fold_scores = {name: [] for name in model_names}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    thr, higher_is_1 = fit_threshold(X_train.ravel(), y_train)
    pred = np.array([classify_threshold(x, thr, higher_is_1) for x in X_test.ravel()])
    model_fold_scores["Threshold"].append(accuracy_score(y_test, pred))

    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    model_fold_scores["Logistic Regression"].append(accuracy_score(y_test, logreg.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42).fit(X_train, y_train)
    model_fold_scores["Random Forest"].append(accuracy_score(y_test, rf.predict(X_test)))

    svm = SVC(kernel="rbf", C=1.0).fit(X_train, y_train)
    model_fold_scores["SVM (RBF)"].append(accuracy_score(y_test, svm.predict(X_test)))

    # DNN: no early stopping (lesson from Task A), average over 5 seeds
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    seed_accs = []
    for seed in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(8,), activation="relu", max_iter=5000,
                             early_stopping=False, random_state=seed, alpha=0.01)
        mlp.fit(X_train_s, y_train)
        seed_accs.append(accuracy_score(y_test, mlp.predict(X_test_s)))
    model_fold_scores["DNN (small MLP)"].append(np.mean(seed_accs))

    if has_xgb:
        xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=42,
                             eval_metric="logloss", verbosity=0).fit(X_train, y_train)
        model_fold_scores["XGBoost"].append(accuracy_score(y_test, xgb.predict(X_test)))

print(f"\n=== {N_FOLDS}-fold GroupKFold cross-validation, Task B (tattoo bit classification) ===\n")
print(f"{'Model':22s} {'Mean Acc':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
summary = {}
for name, scores in model_fold_scores.items():
    scores = np.array(scores)
    summary[name] = scores.mean()
    print(f"{name:22s} {scores.mean()*100:9.2f}% {scores.std()*100:7.2f}% "
          f"{scores.min()*100:7.2f}% {scores.max()*100:7.2f}%")

best = max(summary, key=summary.get)
thr_mean = summary["Threshold"]
print(f"\nBest: {best} ({summary[best]*100:.2f}%), Threshold: {thr_mean*100:.2f}%, "
      f"gap: {(summary[best]-thr_mean)*100:+.2f} points")