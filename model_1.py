import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

X_train = np.load("X_train_rich.npy")
y_train = np.load("y_train_rich.npy")
X_val   = np.load("X_val_rich.npy")
y_val   = np.load("y_val_rich.npy")

print("Rich+ feature shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val:  ", X_val.shape,   "y_val:", y_val.shape)

# Small grid
param_grid = [
    {"learning_rate": 0.1,  "max_depth": 8,  "max_leaf_nodes": 31, "max_iter": 300},
    {"learning_rate": 0.1,  "max_depth": 10, "max_leaf_nodes": 63, "max_iter": 300},
    {"learning_rate": 0.05, "max_depth": 10, "max_leaf_nodes": 63, "max_iter": 400},
    {"learning_rate": 0.1,  "max_depth": None, "max_leaf_nodes": 63, "max_iter": 400},
]

best_acc = 0.0
best_params = None

for params in param_grid:
    print("\nTesting HGB params:", params)
    hgb = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        max_leaf_nodes=params["max_leaf_nodes"],
        max_iter=params["max_iter"],
        random_state=42
    )
    hgb.fit(X_train, y_train)
    y_val_pred = hgb.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print("  Val acc:", acc)
    if acc > best_acc:
        best_acc = acc
        best_params = params

print("\nBest HGB params (val):", best_params)
print("Best val acc (HGB):", best_acc)

# Train RandomForest with fixed params
rf = RandomForestClassifier(
    n_estimators=600,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_val_pred_rf = rf.predict(X_val)
acc_rf = accuracy_score(y_val, y_val_pred_rf)
print("\nRandomForest val acc (rich+):", acc_rf)