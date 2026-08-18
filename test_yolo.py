from ultralytics import YOLO


if __name__ == "__main__":

    MODEL_PATH = r"D:\NEW YOLO\runs\classify\train-5\weights\best.pt"
    DATASET_PATH = r"D:\NEW YOLO\dataset"


    print("=" * 60)
    print("LOADING YOLO MODEL")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    print("Model loaded!")
    print()


    # ======================================================
    # VALIDATION
    # ======================================================

    print("=" * 60)
    print("YOLO VALIDATION")
    print("=" * 60)

    val_results = model.val(
        data=DATASET_PATH,
        split="val",
        imgsz=224,
        workers=0
    )

    print()
    print("YOLO Validation Complete!")
    print(val_results)


    # ======================================================
    # TEST
    # ======================================================

    print()
    print("=" * 60)
    print("YOLO TEST")
    print("=" * 60)

    test_results = model.val(
        data=DATASET_PATH,
        split="test",
        imgsz=224,
        workers=0
    )

    print()
    print("YOLO Test Complete!")
    print(test_results)