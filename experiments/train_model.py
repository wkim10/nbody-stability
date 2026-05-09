import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("Dataset:", X.shape, y.shape)
print("Class distribution in y:", np.bincount(y))
print("First 10 labels:", y[:10])
print("Last 10 labels:", y[-10:])

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True
)

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Model 1: Logistic Regression
# -----------------------------
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# -----------------------------
# Model 2: Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# -----------------------------
# Overfitting / feature diagnostics
# -----------------------------
print("\nTraining accuracy:", rf.score(X_train, y_train))
print("Test accuracy:", rf.score(X_test, y_test))

print("\nMean features for unstable (y=0):")
print(X[y == 0].mean(axis=0))
print("\nMean features for stable (y=1):")
print(X[y == 1].mean(axis=0))

depths = [estimator.get_depth() for estimator in rf.estimators_]
print("\nMean tree depth:", np.mean(depths))
print("Max tree depth:", np.max(depths))

# -----------------------------
# Feature importance (RF)
# -----------------------------
importances = rf.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
# plt.show()