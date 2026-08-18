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

TRAIN_FILE = r"D:\NEW YOLO\yolo_features_train.csv"

VAL_FILE = r"D:\NEW YOLO\yolo_features_val.csv"

TEST_FILE = r"D:\NEW YOLO\yolo_features_test.csv"

SVM_FILE = r"D:\NEW YOLO\svm_model.pkl"

SCALER_FILE = r"D:\NEW YOLO\scaler.pkl"

CONFUSION_MATRIX_FILE = (
    r"D:\NEW YOLO\svm_test_confusion_matrix.png"
)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

print("=" * 60)
print("LOADING YOLO FEATURES")
print("=" * 60)


train_data = pd.read_csv(TRAIN_FILE)

val_data = pd.read_csv(VAL_FILE)

test_data = pd.read_csv(TEST_FILE)


print()
print("Train:", train_data.shape)

print("Validation:", val_data.shape)

print("Test:", test_data.shape)


# ==========================================================
# 3. SEPARATE FEATURES AND LABELS
# ==========================================================

X_train = train_data.drop(
    "label",
    axis=1
)

y_train = train_data["label"]


X_val = val_data.drop(
    "label",
    axis=1
)

y_val = val_data["label"]


X_test = test_data.drop(
    "label",
    axis=1
)

y_test = test_data["label"]


print()
print("Training samples:", len(X_train))

print("Validation samples:", len(X_val))

print("Test samples:", len(X_test))

print()
print("Number of features:", X_train.shape[1])


# ==========================================================
# 4. SCALE FEATURES
# ==========================================================

print()
print("=" * 60)
print("SCALING FEATURES")
print("=" * 60)


scaler = StandardScaler()


# IMPORTANT:
# Fit ONLY on training data

X_train_scaled = scaler.fit_transform(
    X_train
)


# Validation and test use
# the SAME scaler

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


print("Scaling complete!")


# ==========================================================
# 5. CREATE SVM
# ==========================================================

print()
print("=" * 60)
print("CREATING SVM")
print("=" * 60)


svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)


# ==========================================================
# 6. TRAIN SVM
# ==========================================================

print()
print("=" * 60)
print("TRAINING SVM")
print("=" * 60)


svm_model.fit(
    X_train_scaled,
    y_train
)


print("SVM training completed!")


# ==========================================================
# 7. VALIDATION
# ==========================================================

print()
print("=" * 60)
print("SVM VALIDATION")
print("=" * 60)


y_val_pred = svm_model.predict(
    X_val_scaled
)


val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)


print()
print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


print()
print(
    classification_report(
        y_val,
        y_val_pred
    )
)


# ==========================================================
# 8. FINAL TEST
# ==========================================================

print()
print("=" * 60)
print("SVM FINAL TEST")
print("=" * 60)


y_test_pred = svm_model.predict(
    X_test_scaled
)


test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)


print()
print(
    f"FINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)


print()
print(
    classification_report(
        y_test,
        y_test_pred
    )
)


# ==========================================================
# 9. CONFUSION MATRIX
# ==========================================================

classes = sorted(
    y_test.unique()
)


cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=classes
)


print()
print("Confusion Matrix:")
print(cm)


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


plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "SVM Test Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_FILE
)


plt.show()


# ==========================================================
# 10. SAVE SVM
# ==========================================================

joblib.dump(
    svm_model,
    SVM_FILE
)


print()
print("SVM saved:")
print(SVM_FILE)


# ==========================================================
# 11. SAVE SCALER
# ==========================================================

joblib.dump(
    scaler,
    SCALER_FILE
)


print()
print("Scaler saved:")
print(SCALER_FILE)


# ==========================================================
# 12. FINISHED
# ==========================================================

print()
print("=" * 60)
print("SVM TRAINING COMPLETE")
print("=" * 60)

print()
print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)