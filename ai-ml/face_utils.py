# face_utils.py

import os
import logging
import cv2
import torch
from ultralytics import YOLO
from deepface import DeepFace

logger = logging.getLogger(__name__)

# ======================================================
# DEVICE
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[FACE_UTILS] Using device: {DEVICE}")

# ======================================================
# MODEL PATH (LINDEVS)
# ======================================================
FACE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "yolov8n-face-lindevs.pt"
)

if not os.path.exists(FACE_MODEL_PATH):
    raise FileNotFoundError(
        f"YOLOv8-Face (Lindevs) model not found at: {FACE_MODEL_PATH}"
    )

# ======================================================
# LOAD YOLO FACE MODEL (GPU)
# ======================================================
try:
    face_model = YOLO(FACE_MODEL_PATH)
    if DEVICE == "cuda":
        face_model.to("cuda")
    logger.info("YOLOv8-Face (Lindevs) loaded successfully.")
except Exception as e:
    logger.error("Failed to load YOLOv8-Face model", exc_info=True)
    raise

# ======================================================
# OPTIONAL DEEPFACE MODELS
# ======================================================
AVAILABLE_ACTIONS = []
for action in ["Age", "Gender", "Emotion"]:
    try:
        DeepFace.build_model(action)
        AVAILABLE_ACTIONS.append(action.lower())
    except Exception:
        pass

logger.info(f"DeepFace enabled actions: {AVAILABLE_ACTIONS}")

# ======================================================
# MAIN FUNCTION
# ======================================================
def detect_faces(image_path):
    """
    Advanced GPU-based face detection using YOLOv8-Face (Lindevs).
    """

    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0, [], [], [], []

        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_bboxes = []
        face_confidences = []
        face_roles = []
        demographics = []

        # ---------------- YOLO FACE DETECTION ----------------
        results = face_model(image_path, conf=0.4)

        for r in results:
            for box in r.boxes:
                # xywh is center-based
                xc, yc, w, h = box.xywh.tolist()[0]
                conf = float(box.conf.item())

                x = int(xc - w / 2)
                y = int(yc - h / 2)
                w = int(w)
                h = int(h)

                x, y = max(0, x), max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)

                face_bboxes.append([x, y, w, h])
                face_confidences.append(round(conf, 3))

        face_count = len(face_bboxes)

        # ---------------- FACE ROLES (PRIVACY) ----------------
        image_area = img_w * img_h
        for (_, _, w, h) in face_bboxes:
            ratio = (w * h) / image_area

            if ratio > 0.15:
                role = "Primary"
            elif ratio > 0.06:
                role = "Secondary"
            else:
                role = "Background"

            face_roles.append(role)

        # ---------------- DEMOGRAPHIC ANALYSIS ----------------
        for (x, y, w, h) in face_bboxes:
            face_img = img_rgb[y:y+h, x:x+w]

            demo = {
                "age": "Unknown",
                "gender": "Unknown",
                "emotion": "Unknown"
            }

            if AVAILABLE_ACTIONS:
                try:
                    analysis = DeepFace.analyze(
                        face_img,
                        actions=AVAILABLE_ACTIONS,
                        enforce_detection=False
                    )[0]

                    demo["age"] = analysis.get("age", "Unknown")
                    demo["gender"] = analysis.get("dominant_gender", "Unknown")
                    demo["emotion"] = analysis.get("dominant_emotion", "Unknown")

                except Exception:
                    pass

            demographics.append(demo)

        logger.debug(
            f"Faces detected: {face_count}, "
            f"Roles: {face_roles}, "
            f"Confidences: {face_confidences}"
        )

        return (
            face_count,
            face_confidences,
            face_bboxes,
            demographics,
            face_roles
        )

    except Exception:
        logger.error("YOLOv8-Face detection failed", exc_info=True)
        return 0, [], [], [], []
