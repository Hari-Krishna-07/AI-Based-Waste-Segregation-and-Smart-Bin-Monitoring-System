import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


# ==========================================================
# FILES
# ==========================================================

TRAIN_CSV = r"D:\NEW YOLO\yolo_features_train.csv"

VAL_CSV = r"D:\NEW YOLO\yolo_features_val.csv"


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("LOADING FEATURES")
print("=" * 60)

train_data = pd.read_csv(TRAIN_CSV)

val_data = pd.read_csv(VAL_CSV)


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


print()
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Features:", X_train.shape[1])
print()


# ==========================================================
# SCALE
# ==========================================================

print("=" * 60)
print("SCALING")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)


# ==========================================================
# PARAMETERS TO TEST
# ==========================================================

C_VALUES = [
    0.1,
    1,
    10,
    100
]


GAMMA_VALUES = [
    "scale",
    0.001,
    0.01,
    0.1
]


# ==========================================================
# TUNING
# ==========================================================

best_accuracy = 0

best_C = None

best_gamma = None


print()
print("=" * 60)
print("SVM HYPERPARAMETER TUNING")
print("=" * 60)
print()


for C in C_VALUES:

    for gamma in GAMMA_VALUES:

        print(
            f"Testing C={C}, gamma={gamma}"
        )


        svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42
        )


        svm.fit(
            X_train_scaled,
            y_train
        )


        y_val_pred = svm.predict(
            X_val_scaled
        )


        accuracy = accuracy_score(
            y_val,
            y_val_pred
        )


        print(
            f"Validation Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print()


        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_C = C

            best_gamma = gamma


# ==========================================================
# BEST RESULT
# ==========================================================

print("=" * 60)
print("BEST SVM PARAMETERS")
print("=" * 60)

print()

print(
    f"Best C: {best_C}"
)

print(
    f"Best gamma: {best_gamma}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print()