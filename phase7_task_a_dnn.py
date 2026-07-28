import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_pickle("unified_leakage_dataset.pkl")
task_a = df[df["application"] == "counterfeit_detection"].copy()

CATS = ["fresh", "light", "moderate", "severe"]
task_a["label_idx"] = task_a["ground_truth_label"].map({c: i for i, c in enumerate(CATS)})

X = task_a[["leakage_ratio"]].values
y = task_a["label_idx"].values
groups = task_a["chip_id"].values

N_FOLDS = 8
gkf = GroupKFold(n_splits=N_FOLDS)

# Small MLP, appropriate for ~112 training samples / 1 feature -- a large network here
# would just overfit given the data size, so keep it modest and let early stopping guard further.
scores = []
all_seed_results = []
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # No early stopping (too few samples per fold for a reliable internal validation split);
    # run several random seeds since small-data MLPs are sensitive to initialization,
    # and report the mean -- a single seed isn't a fair representation of "what DNNs can do" here.
    seed_accs = []
    for seed in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(8,), activation="relu", max_iter=5000,
                             early_stopping=False, random_state=seed, alpha=0.01)
        mlp.fit(X_train_s, y_train)
        seed_accs.append(accuracy_score(y_test, mlp.predict(X_test_s)))
    fold_acc = np.mean(seed_accs)
    scores.append(fold_acc)
    all_seed_results.append(seed_accs)

scores = np.array(scores)
all_seed_results = np.array(all_seed_results)
print(f"=== DNN (small MLP, 8 hidden units, no early stopping, 5 seeds/fold) -- "
      f"{N_FOLDS}-fold chip-grouped CV ===")
print(f"Mean accuracy: {scores.mean()*100:.1f}%, std: {scores.std()*100:.1f}%, "
      f"min: {scores.min()*100:.1f}%, max: {scores.max()*100:.1f}%")
print(f"Seed-to-seed spread within folds (min/max per fold): "
      f"{all_seed_results.min(axis=1).mean()*100:.1f}% / {all_seed_results.max(axis=1).mean()*100:.1f}%")
print("If seed-to-seed spread is large, the earlier 36.2% result was likely an unlucky")
print("initialization/early-stopping artifact rather than a genuine DNN limitation.")

print(f"\n=== Full Task A comparison (for the paper's Phase 7 table) ===")
print(f"{'Model':22s} {'Mean Acc':>10} {'Std':>8}")
print(f"{'Threshold':22s} {'72.5%':>10} {'3.5%':>8}")
print(f"{'Logistic Regression':22s} {'66.9%':>10} {'2.4%':>8}")
print(f"{'Random Forest':22s} {'70.0%':>10} {'7.9%':>8}")
print(f"{'SVM (RBF)':22s} {'74.4%':>10} {'6.3%':>8}")
print(f"{'XGBoost':22s} {'65.0%':>10} {'9.0%':>8}")
print(f"{'DNN (small MLP)':22s} {scores.mean()*100:9.1f}% {scores.std()*100:7.1f}%")