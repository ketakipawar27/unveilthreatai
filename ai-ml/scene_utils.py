import os
import logging
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ======================================================
# DEVICE SELECTION
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[SCENE_UTILS] Using device: {DEVICE}")

# ======================================================
# MODEL LOADING (GPU-AWARE)
# ======================================================
try:
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    blip_model.eval()

    yolo_model = YOLO("yolov8n.pt")
    if DEVICE == "cuda":
        yolo_model.to("cuda")

except Exception as e:
    logger.error(f"Error loading models in scene_utils: {str(e)}", exc_info=True)
    raise

# ======================================================
# MAIN FUNCTION
# ======================================================
def scene_understanding(image_path):
    """
    Generate scene description using BLIP (GPU)
    and detect objects using YOLO (GPU if available).
    """
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return "", [], []

        # ---------------- BLIP (GPU) ----------------
        img = Image.open(image_path).convert("RGB")

        inputs = blip_processor(
            images=img,
            return_tensors="pt"
        )

        # move tensors to GPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = blip_model.generate(**inputs)

        scene_description = blip_processor.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # ---------------- YOLO (GPU auto) ----------------
        yolo_results = yolo_model(image_path)

        objects = []
        bboxes = []

        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls)
                label = result.names[cls]
                conf = float(box.conf.item())

                if conf > 0.5 and label in [
                    "person", "laptop", "book", "cell phone", "car",
                    "bottle", "key", "handbag", "tie", "wine glass",
                    "suitcase", "keyboard", "mouse", "tv"
                ]:
                    objects.append((label, conf))
                    bboxes.append(box.xywh.tolist()[0])  # [xc, yc, w, h]

        logger.debug(f"Scene: {scene_description}, Objects: {objects}")
        return scene_description, objects, bboxes

    except Exception as e:
        logger.error(
            f"Error in scene understanding for {image_path}: {str(e)}",
            exc_info=True
        )
        return "", [], []
