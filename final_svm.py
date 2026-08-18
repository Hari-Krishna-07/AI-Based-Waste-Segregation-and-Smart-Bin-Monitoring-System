import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# 1. FILE PATHS
# ==========================================================

TRAIN_CSV = r"D:\NEW YOLO\yolo_features_train.csv"

VAL_CSV = r"D:\NEW YOLO\yolo_features_val.csv"

TEST_CSV = r"D:\NEW YOLO\yolo_features_test.csv"

SVM_FILE = r"D:\NEW YOLO\final_svm_model.pkl"

SCALER_FILE = r"D:\NEW YOLO\final_scaler.pkl"

CONFUSION_MATRIX_FILE = (
    r"D:\NEW YOLO\final_svm_confusion_matrix.png"
)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

print()
print("=" * 60)
print("LOADING TRAIN / VALIDATION / TEST FEATURES")
print("=" * 60)
print()

train_data = pd.read_csv(TRAIN_CSV)

val_data = pd.read_csv(VAL_CSV)

test_data = pd.read_csv(TEST_CSV)


print("Train:", train_data.shape)

print("Validation:", val_data.shape)

print("Test:", test_data.shape)

print()


# ==========================================================
# 3. COMBINE TRAIN + VALIDATION
# ==========================================================

print("=" * 60)
print("COMBINING TRAIN + VALIDATION")
print("=" * 60)
print()

combined_train = pd.concat(
    [
        train_data,
        val_data
    ],
    ignore_index=True
)


print(
    "Combined training samples:",
    len(combined_train)
)

print()


# ==========================================================
# 4. SEPARATE FEATURES AND LABELS
# ==========================================================

X_train = combined_train.drop(
    "label",
    axis=1
)

y_train = combined_train["label"]


X_test = test_data.drop(
    "label",
    axis=1
)

y_test = test_data["label"]


print(
    "Training samples:",
    len(X_train)
)

print(
    "Test samples:",
    len(X_test)
)

print(
    "Number of features:",
    X_train.shape[1]
)

print()


# ==========================================================
# 5. CLASS DISTRIBUTION
# ==========================================================

print("=" * 60)
print("FINAL TRAINING CLASS DISTRIBUTION")
print("=" * 60)
print()

print(
    y_train.value_counts()
)

print()


# ==========================================================
# 6. FEATURE SCALING
# ==========================================================

print("=" * 60)
print("SCALING FEATURES")
print("=" * 60)
print()

scaler = StandardScaler()


# IMPORTANT:
# Fit scaler ONLY on final training data

X_train_scaled = scaler.fit_transform(
    X_train
)


# Apply same scaler to test data

X_test_scaled = scaler.transform(
    X_test
)


print("Scaling complete!")

print()


# ==========================================================
# 7. CREATE FINAL SVM
# ==========================================================

print("=" * 60)
print("CREATING FINAL SVM")
print("=" * 60)
print()

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)


print("C = 10")

print("Gamma = scale")

print("Kernel = RBF")

print()


# ==========================================================
# 8. TRAIN FINAL SVM
# ==========================================================

print("=" * 60)
print("TRAINING FINAL SVM")
print("=" * 60)
print()

svm_model.fit(
    X_train_scaled,
    y_train
)


print("Final SVM training completed!")

print()


# ==========================================================
# 9. FINAL TEST
# ==========================================================

print("=" * 60)
print("FINAL SVM TEST")
print("=" * 60)
print()

y_pred = svm_model.predict(
    X_test_scaled
)


# ==========================================================
# 10. ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print()

print(
    f"FINAL SVM TEST ACCURACY: "
    f"{accuracy * 100:.2f}%"
)

print()


# ==========================================================
# 11. CLASSIFICATION REPORT
# ==========================================================

print("=" * 60)
print("FINAL CLASSIFICATION REPORT")
print("=" * 60)
print()

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================================
# 12. CONFUSION MATRIX
# ==========================================================

classes = sorted(
    y_train.unique()
)


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)


print("=" * 60)
print("FINAL CONFUSION MATRIX")
print("=" * 60)
print()

print(cm)

print()


# ==========================================================
# 13. SAVE CONFUSION MATRIX
# ==========================================================

plt.figure(
    figsize=(8, 6)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Final SVM Test Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_FILE
)


plt.show()


print(
    "Confusion matrix saved:"
)

print(
    CONFUSION_MATRIX_FILE
)

print()


# ==========================================================
# 14. SAVE FINAL SVM
# ==========================================================

joblib.dump(
    svm_model,
    SVM_FILE
)


print(
    "Final SVM saved:"
)

print(
    SVM_FILE
)

print()


# ==========================================================
# 15. SAVE FINAL SCALER
# ==========================================================

joblib.dump(
    scaler,
    SCALER_FILE
)


print(
    "Final scaler saved:"
)

print(
    SCALER_FILE
)

print()


# ==========================================================
# 16. COMPLETE
# ==========================================================

print("=" * 60)
print("FINAL SVM TRAINING COMPLETE")
print("=" * 60)
print()