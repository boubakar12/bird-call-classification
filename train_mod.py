import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load 
X_train = np.load("X_train_mfcc.npy")
y_train = np.load("y_train.npy")
X_val   = np.load("X_val_mfcc.npy")
y_val   = np.load("y_val.npy")
print("Shapes:")
print("  X_train:", X_train.shape, " y_train:", y_train.shape)
print("  X_val:  ", X_val.shape,   " y_val:  ", y_val.shape)

# Scale for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# Regression 
logreg = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=-1

)
logreg.fit(X_train_scaled, y_train)
y_val_pred_logreg = logreg.predict(X_val_scaled)
val_acc_logreg = accuracy_score(y_val, y_val_pred_logreg)
print("Validation accuracy (Logistic Regression):", val_acc_logreg)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_val_pred_rf = rf.predict(X_val)
val_acc_rf = accuracy_score(y_val, y_val_pred_rf)
print("Validation accuracy (Random Forest):", val_acc_rf)
