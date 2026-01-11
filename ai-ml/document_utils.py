# document_utils.py

import os
import logging
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

# ===================== DEVICE =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[DOCUMENT_UTILS] Using device: {DEVICE}")

# ===================== LOAD MODEL =====================
try:
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(DEVICE).eval()

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

except Exception as e:
    logger.error(f"Failed to load CLIP in document_utils: {e}", exc_info=True)
    raise

# ===================== DOCUMENT DEFINITIONS =====================

TIER_1_DOCUMENTS = {
    "aadhaar card": "Government ID (India)",
    "pan card": "Tax ID (India)",
    "passport": "Passport",
    "driving license": "Driving License",
    "credit card": "Credit Card",
    "debit card": "Debit Card",
    "bank cheque": "Bank Cheque",
}

TIER_2_DOCUMENTS = {
    "flight ticket": "Flight Ticket",
    "boarding pass": "Boarding Pass",
    "train ticket": "Train Ticket",
    "hotel booking": "Hotel Booking",
    "office id card": "Office ID",
    "college id card": "Student ID",
}

IGNORED_DOCUMENTS = [
    "invoice",
    "contract",
    "certificate",
    "letter",
    "receipt",
    "newspaper",
    "book page"
]

ALL_DOCUMENT_LABELS = (
    list(TIER_1_DOCUMENTS.keys()) +
    list(TIER_2_DOCUMENTS.keys()) +
    IGNORED_DOCUMENTS
)

# ===================== THRESHOLDS =====================
TIER_1_THRESHOLD = 0.35
TIER_2_THRESHOLD = 0.40


def document_inference(image_path):
    """
    Detect ONLY privacy-critical documents using CLIP (GPU-accelerated).
    Returns structured document evidence.
    """

    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            return []

        img = Image.open(image_path).convert("RGB")

        inputs = clip_processor(
            text=ALL_DOCUMENT_LABELS,
            images=img,
            return_tensors="pt",
            padding=True
        )

        # 🔥 MOVE INPUTS TO GPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().numpy()

        detected = []

        for label, prob in zip(ALL_DOCUMENT_LABELS, probs):
            prob = float(prob)

            # ---------------- Tier-1 (ALWAYS HIGH RISK) ----------------
            if label in TIER_1_DOCUMENTS and prob >= TIER_1_THRESHOLD:
                detected.append({
                    "type": "document",
                    "document": TIER_1_DOCUMENTS[label],
                    "raw_label": label,
                    "tier": "HIGH",
                    "confidence": round(prob, 2),
                    "risk_reason": "Identity or financial document"
                })

            # ---------------- Tier-2 (CONDITIONAL RISK) ----------------
            elif label in TIER_2_DOCUMENTS and prob >= TIER_2_THRESHOLD:
                detected.append({
                    "type": "document",
                    "document": TIER_2_DOCUMENTS[label],
                    "raw_label": label,
                    "tier": "MODERATE",
                    "confidence": round(prob, 2),
                    "risk_reason": "Travel or access document"
                })

        logger.debug(f"Document inference result: {detected}")
        return detected

    except Exception as e:
        logger.error(f"Document inference failed: {e}", exc_info=True)
        return []
