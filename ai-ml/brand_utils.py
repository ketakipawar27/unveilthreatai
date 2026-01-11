# brand_utils.py

import os
import logging
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

from ocr_utils import extract_ocr_text, ocr_reader  # unified import

logger = logging.getLogger(__name__)

# ======================================================
# DEVICE SELECTION
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[BRAND_UTILS] Using device: {DEVICE}")

# ======================================================
# LOAD MODELS (ONCE, GPU-AWARE)
# ======================================================
try:
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(DEVICE)
    clip_model.eval()

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    yolo_model = YOLO("yolov8n.pt")
    if DEVICE == "cuda":
        yolo_model.to("cuda")

    logger.info("Brand models loaded successfully.")

except Exception as e:
    logger.error(
        f"Error loading models in brand_utils: {str(e)}",
        exc_info=True
    )
    raise

# ======================================================
# BRAND INFERENCE
# ======================================================
def brand_inference(image_path):
    """
    Brand detection using:
    1) CLIP (visual similarity) → GPU
    2) OCR (brand text) → CPU
    3) YOLO (logo objects) → GPU

    Returns:
    - detected_brands: [(brand, confidence, source)]
    - brand_bboxes: [{type, bbox, label}]
    """

    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return [], []

        # --------------------------------------------------
        # LOAD IMAGE
        # --------------------------------------------------
        img_pil = Image.open(image_path).convert("RGB")
        img_cv = cv2.imread(image_path)

        if img_cv is None:
            logger.error(f"Failed to load image for brand inference: {image_path}")
            return [], []

        # --------------------------------------------------
        # BRAND LIST
        # --------------------------------------------------
        brands = [
            "Apple", "Google", "Microsoft", "Samsung",
            "Nike", "Adidas", "Puma",
            "Coca Cola", "Pepsi",
            "BMW", "Mercedes",
            "Louis Vuitton", "Gucci", "Prada",
            "Rolex"
        ]

        detected_brands = []
        seen = set()
        brand_bboxes = []

        # ==================================================
        # 1️⃣ CLIP BRAND MATCHING (GPU)
        # ==================================================
        try:
            inputs = clip_processor(
                text=brands,
                images=img_pil,
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = clip_model(**inputs)

            probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

            for brand, prob in zip(brands, probs):
                if prob >= 0.70 and brand not in seen:
                    detected_brands.append((brand, round(prob, 3), "CLIP"))
                    seen.add(brand)

        except Exception as e:
            logger.warning(f"CLIP brand inference failed: {e}")

        # ==================================================
        # 2️⃣ OCR BRAND TEXT MATCHING (CPU)
        # ==================================================
        try:
            extract_ocr_text(image_path)  # ensures OCR cache consistency

            if ocr_reader:
                ocr_results = ocr_reader.readtext(img_cv, detail=1)

                for bbox, text, conf in ocr_results:
                    if conf < 0.7:
                        continue

                    for brand in brands:
                        if brand.lower() in text.lower() and brand not in seen:
                            detected_brands.append(
                                (brand, round(float(conf), 3), "OCR")
                            )
                            seen.add(brand)

                            xs = [p[0] for p in bbox]
                            ys = [p[1] for p in bbox]
                            x, y = min(xs), min(ys)
                            w, h = max(xs) - x, max(ys) - y

                            brand_bboxes.append({
                                "type": "brand_text",
                                "bbox": [x, y, w, h],
                                "label": f"Brand: {brand}"
                            })

        except Exception as e:
            logger.warning(f"OCR brand inference failed: {e}")

        # ==================================================
        # 3️⃣ YOLO LOGO DETECTION (GPU)
        # ==================================================
        try:
            yolo_results = yolo_model(image_path)

            for result in yolo_results:
                for box in result.boxes:
                    cls = int(box.cls)
                    label = result.names.get(cls, "").lower()
                    conf = float(box.conf.item())

                    if conf < 0.7:
                        continue

                    for brand in brands:
                        if brand.lower() in label and brand not in seen:
                            detected_brands.append(
                                (brand, round(conf, 3), "Logo")
                            )
                            seen.add(brand)

                            x, y, w, h = box.xywh.tolist()[0]
                            brand_bboxes.append({
                                "type": "brand_logo",
                                "bbox": [x, y, w, h],
                                "label": f"Brand Logo: {brand}"
                            })

        except Exception:
            logger.info("YOLO logo detection skipped (no logo-specific model).")

        logger.debug(
            f"Brand inference result: {detected_brands}, "
            f"BBoxes: {brand_bboxes}"
        )

        return detected_brands, brand_bboxes

    except Exception as e:
        logger.error(
            f"Error in brand inference for {image_path}: {str(e)}",
            exc_info=True
        )
        return [], []
