from ultralytics import YOLO
import cv2
import numpy as np
import joblib
from collections import Counter
import time
import serial


# ==========================================================
# 1. MODEL SETTINGS
# ==========================================================

YOLO_MODEL = r"D:\NEW YOLO\runs\classify\train-11\weights\best.pt"

SVM_MODEL = r"D:\NEW YOLO\final_svm_model.pkl"

SCALER = r"D:\NEW YOLO\final_scaler.pkl"


# ==========================================================
# 2. ESP32 SERIAL SETTINGS
# ==========================================================

ESP32_PORT = "COM5"

ESP32_BAUD = 115200


# ==========================================================
# 3. CAMERA SETTINGS
# ==========================================================

FRAME_WIDTH = 640

FRAME_HEIGHT = 480


# ==========================================================
# 4. DETECTION AREA
# ==========================================================

ROI_X1 = 220
ROI_Y1 = 120

ROI_X2 = 420
ROI_Y2 = 350


# ==========================================================
# 5. OBJECT PLACEMENT AREA
# ==========================================================

OBJECT_X1 = 220
OBJECT_Y1 = 180

OBJECT_X2 = 420
OBJECT_Y2 = 390


# ==========================================================
# 6. OBJECT DETECTION SETTINGS
# ==========================================================

MIN_AREA = 2500

DIFF_THRESHOLD = 30


# ==========================================================
# 7. CLASSIFICATION SETTINGS
# ==========================================================

TOTAL_FRAMES = 10

SVM_CONFIDENCE_THRESHOLD = 0.75

FRAME_DELAY = 0.15


# ==========================================================
# 8. OBJECT REMOVAL SETTINGS
# ==========================================================

EMPTY_FRAMES_REQUIRED = 15


# ==========================================================
# 9. YOLO EMBEDDING FEATURES
# ==========================================================

EXPECTED_FEATURES = 256


# ==========================================================
# 10. VALID WASTE TYPES
# ==========================================================

VALID_WASTE = {
    "PAPER",
    "PLASTIC",
    "METAL",
    "GLASS"
}


# ==========================================================
# 11. CONNECT ESP32
# ==========================================================

print()
print("=" * 60)
print("CONNECTING TO ESP32")
print("=" * 60)
print()

try:

    esp32 = serial.Serial(
        ESP32_PORT,
        ESP32_BAUD,
        timeout=1
    )

    time.sleep(2)

    esp32.reset_input_buffer()

    print("ESP32 CONNECTED")
    print("PORT:", ESP32_PORT)
    print("BAUD:", ESP32_BAUD)

except Exception as e:

    print()
    print("ERROR: ESP32 COULD NOT BE CONNECTED")
    print(e)
    print()
    print("CHECK:")
    print("1. ESP32 USB cable")
    print("2. COM port")
    print("3. Arduino Serial Monitor is CLOSED")
    print()

    exit()


# ==========================================================
# 12. LOAD YOLO MODEL
# ==========================================================

print()
print("=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)
print()

print("MODEL:")
print(YOLO_MODEL)
print()

try:

    model = YOLO(YOLO_MODEL)

except Exception as e:

    print("ERROR: Could not load YOLO model.")
    print(e)

    esp32.close()

    exit()


print()
print("YOLO MODEL LOADED SUCCESSFULLY")
print()

print("YOLO CLASSES:")

for class_id, class_name in model.names.items():

    print(
        f"   {class_id} : {class_name}"
    )


# ==========================================================
# 13. LOAD SVM
# ==========================================================

print()
print("=" * 60)
print("LOADING SVM MODEL")
print("=" * 60)
print()

print("SVM:")
print(SVM_MODEL)
print()

try:

    svm_model = joblib.load(
        SVM_MODEL
    )

except Exception as e:

    print("ERROR: Could not load SVM model.")
    print(e)

    esp32.close()

    exit()


print("SVM MODEL LOADED SUCCESSFULLY")


# ==========================================================
# 14. LOAD SCALER
# ==========================================================

print()
print("=" * 60)
print("LOADING SCALER")
print("=" * 60)
print()

print("SCALER:")
print(SCALER)
print()

try:

    scaler = joblib.load(
        SCALER
    )

except Exception as e:

    print("ERROR: Could not load scaler.")
    print(e)

    esp32.close()

    exit()


print("SCALER LOADED SUCCESSFULLY")

print()

print(
    "Scaler features:",
    scaler.n_features_in_
)

print(
    "Expected features:",
    EXPECTED_FEATURES
)


# ==========================================================
# 15. DISPLAY SVM CLASSES
# ==========================================================

print()
print("SVM CLASSES:")

for class_name in svm_model.classes_:

    print(
        "   ",
        class_name
    )


# ==========================================================
# 16. CHECK FEATURE COUNT
# ==========================================================

if scaler.n_features_in_ != EXPECTED_FEATURES:

    print()
    print("=" * 60)
    print("WARNING")
    print("=" * 60)

    print(
        "Scaler does not expect",
        EXPECTED_FEATURES,
        "features."
    )

    print(
        "Scaler expects:",
        scaler.n_features_in_
    )

    print()


# ==========================================================
# 17. CHECK SVM PROBABILITY
# ==========================================================

if not hasattr(
    svm_model,
    "predict_proba"
):

    print()
    print("ERROR!")
    print("SVM does not support probability prediction.")
    print("Train SVM with probability=True.")

    esp32.close()

    exit()


# ==========================================================
# 18. SEND COMMAND TO ESP32
# ==========================================================

def send_to_esp32(
    waste_type
):

    try:

        command = str(
            waste_type
        ).strip().upper()


        # ==================================================
        # SAFETY CHECK
        # ==================================================

        if command not in VALID_WASTE:

            print()
            print("=" * 60)
            print("SAFETY BLOCK")
            print("=" * 60)

            print(
                "Invalid command:",
                command
            )

            print(
                "NOT SENT TO ESP32"
            )

            print(
                "TOP SERVO WILL NOT OPEN"
            )

            print("=" * 60)

            return False


        print()
        print("=" * 60)
        print("SENDING TO ESP32")
        print("=" * 60)

        print(
            "VALID COMMAND:",
            command
        )


        esp32.write(
            (
                command + "\n"
            ).encode()
        )

        esp32.flush()


        print(
            "SENT SUCCESSFULLY"
        )

        print("=" * 60)


        return True


    except Exception as e:

        print()
        print("ERROR SENDING TO ESP32:")
        print(e)

        return False


# ==========================================================
# 19. OPEN CAMERA
# ==========================================================

print()
print("=" * 60)
print("OPENING CAMERA")
print("=" * 60)
print()

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


if not cap.isOpened():

    print(
        "ERROR: CAMERA COULD NOT BE OPENED"
    )

    esp32.close()

    exit()


print("CAMERA STARTED")


# ==========================================================
# 20. CAPTURE EMPTY BACKGROUND
# ==========================================================

print()
print("=" * 60)
print("CAPTURING EMPTY BACKGROUND")
print("=" * 60)
print()

print(
    "REMOVE ALL OBJECTS FROM THE YELLOW BOX."
)

print()

background_frames = []


for i in range(40):

    ret, frame = cap.read()

    if not ret:

        continue


    frame = cv2.resize(
        frame,
        (
            FRAME_WIDTH,
            FRAME_HEIGHT
        )
    )


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    background_frames.append(
        gray
    )


    display = frame.copy()


    cv2.rectangle(
        display,
        (
            OBJECT_X1,
            OBJECT_Y1
        ),
        (
            OBJECT_X2,
            OBJECT_Y2
        ),
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Capturing background {i + 1}/40",
        (
            20,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    cv2.imshow(
        "Waste Detection",
        display
    )


    cv2.waitKey(50)


# ==========================================================
# 21. CHECK BACKGROUND
# ==========================================================

if len(background_frames) == 0:

    print(
        "ERROR: Could not capture background."
    )

    cap.release()

    esp32.close()

    cv2.destroyAllWindows()

    exit()


# ==========================================================
# 22. CREATE BACKGROUND
# ==========================================================

background = np.median(
    np.array(
        background_frames
    ),
    axis=0
).astype(
    np.uint8
)


print()
print("BACKGROUND CAPTURED SUCCESSFULLY")
print()

print("READY")

print(
    "Place ONE waste object inside the yellow box."
)

print()


# ==========================================================
# 23. SYSTEM STATES
# ==========================================================

WAITING_FOR_OBJECT = 0

CAPTURING_FRAMES = 1

WAITING_FOR_REMOVAL = 2


state = WAITING_FOR_OBJECT


# ==========================================================
# 24. CLASSIFICATION VARIABLES
# ==========================================================

frame_predictions = []

frame_confidences = []

frames_captured = 0

final_class = None

final_confidence = 0.0

empty_frame_count = 0


# ==========================================================
# 25. DETECT OBJECT
# ==========================================================

def detect_object(
    frame,
    background
):


    # ======================================================
    # GET ROI
    # ======================================================

    roi = frame[
        ROI_Y1:ROI_Y2,
        ROI_X1:ROI_X2
    ]


    # ======================================================
    # GRAYSCALE
    # ======================================================

    roi_gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================================
    # BLUR
    # ======================================================

    roi_gray = cv2.GaussianBlur(
        roi_gray,
        (5, 5),
        0
    )


    # ======================================================
    # BACKGROUND ROI
    # ======================================================

    background_roi = background[
        ROI_Y1:ROI_Y2,
        ROI_X1:ROI_X2
    ]


    # ======================================================
    # DIFFERENCE
    # ======================================================

    difference = cv2.absdiff(
        roi_gray,
        background_roi
    )


    # ======================================================
    # THRESHOLD
    # ======================================================

    _, mask = cv2.threshold(
        difference,
        DIFF_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )


    # ======================================================
    # MORPHOLOGY
    # ======================================================

    kernel = np.ones(
        (7, 7),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2
    )


    # ======================================================
    # FIND CONTOURS
    # ======================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    best_contour = None

    largest_area = 0


    # ======================================================
    # FIND LARGEST OBJECT
    # ======================================================

    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area < MIN_AREA:

            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        center_x = (
            ROI_X1
            + x
            + w // 2
        )


        center_y = (
            ROI_Y1
            + y
            + h // 2
        )


        if not (
            OBJECT_X1 <= center_x <= OBJECT_X2
            and
            OBJECT_Y1 <= center_y <= OBJECT_Y2
        ):

            continue


        if area > largest_area:

            largest_area = area

            best_contour = contour


    # ======================================================
    # NO OBJECT
    # ======================================================

    if best_contour is None:

        return False, None


    # ======================================================
    # OBJECT BOX
    # ======================================================

    x, y, w, h = cv2.boundingRect(
        best_contour
    )


    bx1 = ROI_X1 + x

    by1 = ROI_Y1 + y

    bx2 = ROI_X1 + x + w

    by2 = ROI_Y1 + y + h


    bx1 = max(
        ROI_X1,
        bx1
    )

    by1 = max(
        ROI_Y1,
        by1
    )

    bx2 = min(
        ROI_X2,
        bx2
    )

    by2 = min(
        ROI_Y2,
        by2
    )


    if (
        bx2 <= bx1
        or
        by2 <= by1
    ):

        return False, None


    return True, (
        bx1,
        by1,
        bx2,
        by2
    )


# ==========================================================
# 26. YOLO + SVM CLASSIFICATION
# ==========================================================

def classify_object(
    frame,
    box
):


    bx1, by1, bx2, by2 = box


    # ======================================================
    # CROP
    # ======================================================

    object_crop = frame[
        by1:by2,
        bx1:bx2
    ]


    if object_crop.size == 0:

        return None, 0.0


    crop_height, crop_width = (
        object_crop.shape[:2]
    )


    if (
        crop_width < 20
        or
        crop_height < 20
    ):

        return None, 0.0


    try:

        # ==================================================
        # YOLO EMBEDDING
        # ==================================================

        embeddings = model.embed(
            source=object_crop,
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


        feature_count = embedding.shape[0]


        if feature_count != EXPECTED_FEATURES:

            print()
            print(
                "ERROR: WRONG YOLO EMBEDDING SIZE"
            )

            print(
                "Expected:",
                EXPECTED_FEATURES
            )

            print(
                "Received:",
                feature_count
            )

            return None, 0.0


        # ==================================================
        # RESHAPE
        # ==================================================

        embedding = embedding.reshape(
            1,
            -1
        )


        # ==================================================
        # SCALE
        # ==================================================

        embedding_scaled = scaler.transform(
            embedding
        )


        # ==================================================
        # SVM PROBABILITIES
        # ==================================================

        probabilities = svm_model.predict_proba(
            embedding_scaled
        )


        # ==================================================
        # PRINT PROBABILITIES
        # ==================================================

        print()
        print("SVM PROBABILITIES:")


        for class_name, probability in zip(
            svm_model.classes_,
            probabilities[0]
        ):

            print(
                f"   {class_name}: "
                f"{probability * 100:.2f}%"
            )


        # ==================================================
        # HIGHEST PROBABILITY
        # ==================================================

        class_index = np.argmax(
            probabilities[0]
        )


        predicted_class = (
            svm_model.classes_[class_index]
        )


        confidence = float(
            probabilities[0][class_index]
        )


        # ==================================================
        # NORMALIZE CLASS NAME
        # ==================================================

        predicted_class = str(
            predicted_class
        ).strip().upper()


        # ==================================================
        # CONFIDENCE CHECK
        # ==================================================

        if confidence < SVM_CONFIDENCE_THRESHOLD:

            print()
            print(
                "LOW CONFIDENCE"
            )

            print(
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            )

            print(
                "This frame will NOT be counted."
            )

            return None, confidence


        # ==================================================
        # VALID CLASS CHECK
        # ==================================================

        if predicted_class not in VALID_WASTE:

            print()
            print(
                "INVALID CLASS:"
            )

            print(
                predicted_class
            )

            print(
                "This frame will NOT be counted."
            )

            return None, confidence


        # ==================================================
        # RETURN VALID RESULT
        # ==================================================

        return (
            predicted_class,
            confidence
        )


    except Exception as e:

        print()
        print(
            "ERROR DURING YOLO -> SVM:"
        )

        print(e)

        return None, 0.0


# ==========================================================
# 27. MAIN CAMERA LOOP
# ==========================================================

try:

    while True:


        # ==================================================
        # READ CAMERA
        # ==================================================

        ret, frame = cap.read()


        if not ret:

            print(
                "ERROR: Could not read camera."
            )

            break


        # ==================================================
        # RESIZE
        # ==================================================

        frame = cv2.resize(
            frame,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            )
        )


        display = frame.copy()


        # ==================================================
        # DETECT OBJECT
        # ==================================================

        object_detected, object_box = (
            detect_object(
                frame,
                background
            )
        )


        # ==================================================
        # STATE 1
        # WAITING FOR OBJECT
        # ==================================================

        if state == WAITING_FOR_OBJECT:


            cv2.rectangle(
                display,
                (
                    OBJECT_X1,
                    OBJECT_Y1
                ),
                (
                    OBJECT_X2,
                    OBJECT_Y2
                ),
                (0, 255, 255),
                3
            )


            if not object_detected:

                cv2.putText(
                    display,
                    "PLACE OBJECT HERE",
                    (
                        OBJECT_X1,
                        OBJECT_Y1 - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


            else:

                print()
                print("=" * 60)
                print("OBJECT DETECTED")
                print("=" * 60)

                print(
                    "Starting 10-frame classification..."
                )


                frame_predictions = []

                frame_confidences = []

                frames_captured = 0


                state = CAPTURING_FRAMES


        # ==================================================
        # STATE 2
        # CAPTURE FRAMES
        # ==================================================

        elif state == CAPTURING_FRAMES:


            if not object_detected:

                print()
                print(
                    "OBJECT DISAPPEARED"
                )

                print(
                    "Restarting..."
                )


                frame_predictions = []

                frame_confidences = []

                frames_captured = 0


                state = WAITING_FOR_OBJECT


            else:


                cv2.rectangle(
                    display,
                    (
                        OBJECT_X1,
                        OBJECT_Y1
                    ),
                    (
                        OBJECT_X2,
                        OBJECT_Y2
                    ),
                    (0, 255, 255),
                    3
                )


                predicted_class, confidence = (
                    classify_object(
                        frame,
                        object_box
                    )
                )


                if predicted_class is not None:


                    frame_predictions.append(
                        predicted_class
                    )


                    frame_confidences.append(
                        confidence
                    )


                    frames_captured += 1


                    print(
                        f"Frame "
                        f"{frames_captured}/"
                        f"{TOTAL_FRAMES}"
                        f" -> "
                        f"{predicted_class}"
                        f" "
                        f"{confidence * 100:.2f}%"
                    )


                    cv2.putText(
                        display,
                        f"FRAME "
                        f"{frames_captured}/"
                        f"{TOTAL_FRAMES}",
                        (
                            OBJECT_X1,
                            OBJECT_Y1 - 40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2
                    )


                    cv2.putText(
                        display,
                        f"{predicted_class} "
                        f"{confidence * 100:.1f}%",
                        (
                            OBJECT_X1,
                            OBJECT_Y1 - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )


                    # ======================================
                    # 10 VALID FRAMES COMPLETE
                    # ======================================

                    if frames_captured >= TOTAL_FRAMES:


                        print()
                        print("=" * 60)
                        print("10 FRAMES COMPLETED")
                        print("=" * 60)


                        # ==================================
                        # COUNT
                        # ==================================

                        class_counts = Counter(
                            frame_predictions
                        )


                        print()
                        print(
                            "PREDICTION COUNT:"
                        )


                        for class_name, count in (
                            class_counts.most_common()
                        ):

                            print(
                                f"{class_name} = {count}"
                            )


                        # ==================================
                        # MAJORITY
                        # ==================================

                        final_class, final_votes = (
                            class_counts.most_common(1)[0]
                        )


                        final_confidence = float(
                            np.mean(
                                frame_confidences
                            )
                        )


                        # ==================================
                        # FINAL SAFETY CHECK
                        # ==================================

                        if final_votes < 6:

                            final_class = "UNKNOWN"


                            print()
                            print(
                                "FINAL RESULT: UNKNOWN"
                            )

                            print(
                                "Reason: No 6/10 majority."
                            )


                        elif (
                            final_confidence
                            <
                            SVM_CONFIDENCE_THRESHOLD
                        ):

                            final_class = "UNKNOWN"


                            print()
                            print(
                                "FINAL RESULT: UNKNOWN"
                            )

                            print(
                                "Reason: Average confidence too low."
                            )


                        elif final_class not in VALID_WASTE:

                            final_class = "UNKNOWN"


                            print()
                            print(
                                "FINAL RESULT: UNKNOWN"
                            )

                            print(
                                "Reason: Invalid class."
                            )


                        # ==================================
                        # FINAL RESULT
                        # ==================================

                        print()
                        print("=" * 60)
                        print("FINAL DETECTION")
                        print("=" * 60)


                        print(
                            "Waste:",
                            final_class
                        )


                        print(
                            f"Votes: "
                            f"{final_votes}/"
                            f"{TOTAL_FRAMES}"
                        )


                        print(
                            f"Average confidence: "
                            f"{final_confidence * 100:.2f}%"
                        )


                        print()


                        # ==================================
                        # SEND ONLY VALID WASTE
                        # ==================================

                        if (
                            final_class in VALID_WASTE
                        ):

                            print(
                                "VALID WASTE"
                            )

                            print(
                                "Sending to ESP32..."
                            )


                            send_to_esp32(
                                final_class
                            )


                        else:

                            print()
                            print("=" * 60)
                            print("UNKNOWN WASTE")
                            print("=" * 60)

                            print(
                                "NOT SENT TO ESP32"
                            )

                            print(
                                "TOP SERVO WILL NOT OPEN"
                            )

                            print("=" * 60)


                        print()


                        # ==================================
                        # WAIT FOR REMOVAL
                        # ==================================

                        state = WAITING_FOR_REMOVAL

                        empty_frame_count = 0


                    time.sleep(
                        FRAME_DELAY
                    )


                else:

                    cv2.putText(
                        display,
                        "LOW CONFIDENCE",
                        (
                            20,
                            35
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )


        # ==================================================
        # STATE 3
        # WAIT FOR OBJECT REMOVAL
        # ==================================================

        elif state == WAITING_FOR_REMOVAL:


            if object_detected:

                empty_frame_count = 0


                cv2.rectangle(
                    display,
                    (
                        OBJECT_X1,
                        OBJECT_Y1
                    ),
                    (
                        OBJECT_X2,
                        OBJECT_Y2
                    ),
                    (0, 255, 0),
                    3
                )


                cv2.putText(
                    display,
                    f"FINAL: "
                    f"{final_class}",
                    (
                        OBJECT_X1,
                        OBJECT_Y1 - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    display,
                    "REMOVE OBJECT",
                    (
                        OBJECT_X1,
                        OBJECT_Y2 + 60
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


            else:

                empty_frame_count += 1


                cv2.putText(
                    display,
                    "OBJECT REMOVED",
                    (
                        OBJECT_X1,
                        OBJECT_Y1 - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


                cv2.putText(
                    display,
                    f"WAITING... "
                    f"{empty_frame_count}/"
                    f"{EMPTY_FRAMES_REQUIRED}",
                    (
                        OBJECT_X1,
                        OBJECT_Y2 + 30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


                if (
                    empty_frame_count
                    >=
                    EMPTY_FRAMES_REQUIRED
                ):


                    print()
                    print("=" * 60)
                    print("OBJECT REMOVED")
                    print("=" * 60)

                    print(
                        "READY FOR NEXT WASTE"
                    )

                    print()


                    frame_predictions = []

                    frame_confidences = []

                    frames_captured = 0

                    final_class = None

                    final_confidence = 0.0

                    empty_frame_count = 0

                    state = WAITING_FOR_OBJECT


        # ==================================================
        # DISPLAY
        # ==================================================

        cv2.imshow(
            "Waste Detection",
            display
        )


        # ==================================================
        # KEYBOARD
        # ==================================================

        key = cv2.waitKey(1) & 0xFF


        # ==================================================
        # Q = QUIT
        # ==================================================

        if key == ord("q"):

            print()
            print("EXITING...")

            break


        # ==================================================
        # R = RESET BACKGROUND
        # ==================================================

        if key == ord("r"):


            print()
            print("=" * 60)
            print("RESETTING BACKGROUND")
            print("=" * 60)

            print(
                "REMOVE ALL OBJECTS."
            )

            print(
                "Capturing new background..."
            )

            print()


            background_frames = []


            for i in range(40):


                ret, new_frame = (
                    cap.read()
                )


                if not ret:

                    continue


                new_frame = cv2.resize(
                    new_frame,
                    (
                        FRAME_WIDTH,
                        FRAME_HEIGHT
                    )
                )


                new_gray = cv2.cvtColor(
                    new_frame,
                    cv2.COLOR_BGR2GRAY
                )


                new_gray = cv2.GaussianBlur(
                    new_gray,
                    (5, 5),
                    0
                )


                background_frames.append(
                    new_gray
                )


                reset_display = (
                    new_frame.copy()
                )


                cv2.rectangle(
                    reset_display,
                    (
                        OBJECT_X1,
                        OBJECT_Y1
                    ),
                    (
                        OBJECT_X2,
                        OBJECT_Y2
                    ),
                    (0, 255, 255),
                    2
                )


                cv2.putText(
                    reset_display,
                    f"Resetting background "
                    f"{i + 1}/40",
                    (
                        20,
                        35
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )


                cv2.imshow(
                    "Waste Detection",
                    reset_display
                )


                cv2.waitKey(50)


            if len(background_frames) > 0:

                background = np.median(
                    np.array(
                        background_frames
                    ),
                    axis=0
                ).astype(
                    np.uint8
                )


            frame_predictions = []

            frame_confidences = []

            frames_captured = 0

            final_class = None

            final_confidence = 0.0

            empty_frame_count = 0

            state = WAITING_FOR_OBJECT


            print()
            print(
                "BACKGROUND RESET SUCCESSFULLY"
            )

            print(
                "READY FOR NEXT WASTE"
            )

            print()


# ==========================================================
# 28. CLOSE EVERYTHING
# ==========================================================

except KeyboardInterrupt:

    print()
    print(
        "PROGRAM STOPPED"
    )


finally:

    cap.release()

    cv2.destroyAllWindows()


    try:

        esp32.close()

    except:

        pass


    print()
    print("=" * 60)
    print("SYSTEM CLOSED")
    print("=" * 60)