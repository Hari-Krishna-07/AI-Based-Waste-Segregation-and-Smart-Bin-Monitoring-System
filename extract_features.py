from ultralytics import YOLO
from pathlib import Path
import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = r"D:\NEW YOLO\runs\classify\train-11\weights\best.pt"

TRAIN_DIR = Path(r"D:\NEW YOLO\dataset\train")

VAL_DIR = Path(r"D:\NEW YOLO\dataset\val")

TEST_DIR = Path(r"D:\NEW YOLO\dataset\test")

TRAIN_OUTPUT = r"D:\NEW YOLO\yolo_features_train.csv"

VAL_OUTPUT = r"D:\NEW YOLO\yolo_features_val.csv"

TEST_OUTPUT = r"D:\NEW YOLO\yolo_features_test.csv"


# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================================
# LOAD YOLO MODEL
# ==========================================================

print("=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print()


# ==========================================================
# FEATURE EXTRACTION FUNCTION
# ==========================================================

def extract_features(dataset_dir, output_file):

    features_list = []
    labels_list = []

    print("=" * 60)
    print("EXTRACTING FEATURES")
    print("=" * 60)

    print("Dataset:", dataset_dir)
    print()


    for class_folder in sorted(dataset_dir.iterdir()):

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        print()
        print("Class:", class_name)


        image_files = [
            p for p in class_folder.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print("Images:", len(image_files))


        for index, image_path in enumerate(image_files):

            try:

                embeddings = model.embed(
                    source=str(image_path),
                    imgsz=224,
                    verbose=False
                )


                embedding = (
                    embeddings[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )


                features_list.append(
                    embedding
                )

                labels_list.append(
                    class_name
                )


                if (index + 1) % 50 == 0:

                    print(
                        f"Processed "
                        f"{index + 1}/{len(image_files)}"
                    )


            except Exception as e:

                print(
                    "ERROR processing:",
                    image_path
                )

                print(e)


    # ======================================================
    # CREATE DATASET
    # ======================================================

    X = np.array(features_list)

    y = np.array(labels_list)


    print()
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Classes:", sorted(set(y)))


    feature_columns = [
        f"feature_{i}"
        for i in range(X.shape[1])
    ]


    df = pd.DataFrame(
        X,
        columns=feature_columns
    )


    df["label"] = y


    # ======================================================
    # SAVE
    # ======================================================

    df.to_csv(
        output_file,
        index=False
    )


    print()
    print("Saved:")
    print(output_file)

    print()
    print("Class distribution:")

    print(
        df["label"].value_counts()
    )

    print()


# ==========================================================
# TRAIN
# ==========================================================

print()
print("========== TRAIN ==========")

extract_features(
    TRAIN_DIR,
    TRAIN_OUTPUT
)


# ==========================================================
# VALIDATION
# ==========================================================

print()
print("========== VALIDATION ==========")

extract_features(
    VAL_DIR,
    VAL_OUTPUT
)


# ==========================================================
# TEST
# ==========================================================

print()
print("========== TEST ==========")

extract_features(
    TEST_DIR,
    TEST_OUTPUT
)


print()
print("=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)