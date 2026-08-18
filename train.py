from ultralytics import YOLO


if __name__ == "__main__":

    # Load pretrained YOLOv8 classification model
    model = YOLO("yolov8n-cls.pt")

    # Train using TRAIN images
    # TEST folder will be used as validation by this setup
    model.train(
        data=r"D:\NEW YOLO\dataset",
        epochs=100,
        imgsz=224,
        batch=16,
        device=0,
        workers=0
    )