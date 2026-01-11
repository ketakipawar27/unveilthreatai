# ocr_utils.py

import os
import logging
import cv2
import re
import spacy
import easyocr

logger = logging.getLogger(__name__)

# ===================== INIT =====================
try:
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"EasyOCR init failed: {e}")
    ocr_reader = None

nlp = spacy.load("en_core_web_sm")

# ===================== OCR NOISE FILTER CONFIG =====================
STOCK_WATERMARKS = {
    "alamy", "istock", "shutterstock", "getty",
    "depositphotos", "dreamstime", "123rf"
}

GENERIC_WORDS = {
    "www", "http", "https", "image", "photo",
    "stock", "com", "id", "img"
}

MIN_TEXT_LENGTH = 4


def is_ocr_noise(text: str) -> bool:
    """High-precision OCR noise suppression."""
    if not text:
        return True

    t = text.strip().lower()

    if len(t) < MIN_TEXT_LENGTH:
        return True

    # Must contain letters
    if not any(c.isalpha() for c in t):
        return True

    # Stock watermarks
    if any(w in t for w in STOCK_WATERMARKS):
        return True

    # Generic junk
    if t in GENERIC_WORDS:
        return True

    # OCR garbage heuristic (vowel ratio)
    letters = [c for c in t if c.isalpha()]
    if letters:
        vowel_ratio = sum(c in "aeiou" for c in letters) / len(letters)
        if vowel_ratio < 0.25:
            return True

    return False


# ===================== DOCUMENT KEYWORDS =====================
DOCUMENT_KEYWORDS = {
    "AADHAAR": [
        "aadhaar", "uidai", "unique identification", "government of india"
    ],
    "PAN": [
        "pan", "permanent account number", "income tax department", "epan"
    ],
    "PASSPORT": [
        "passport", "republic of india", "place of issue", "nationality"
    ],
    "BANK_CARD": [
        "credit card", "debit card", "valid thru", "expiry", "cvv"
    ],
    "BANK_ACCOUNT": [
        "account number", "ifsc", "bank", "branch", "statement"
    ],
    "TICKET": [
        "ticket", "boarding pass", "pnr", "flight", "train"
    ]
}

# ===================== REGEX PATTERNS (HIGH CONFIDENCE) =====================
PATTERNS = {
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "PHONE": r"(\+91|0)?[ -]?\d{10}",
    "PINCODE": r"\b\d{6}\b",
    "CARD_NUMBER": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "LICENSE_PLATE": r"\b[A-Z]{2}\d{2}[A-Z]{2}\d{4}\b"
}

# ===================== MAIN FUNCTION =====================
def extract_ocr_text(image_path):
    """
    Extract OCR text and detect:
    - Sensitive PII
    - Semantic document indicators
    """

    if not ocr_reader:
        return "", [], []

    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            return "", [], []

        img = cv2.imread(image_path)
        if img is None:
            return "", [], []

        # ---------- OCR ----------
        raw_results = ocr_reader.readtext(img, detail=0)

        clean_texts = [
            t for t in raw_results if not is_ocr_noise(t)
        ]

        text = " ".join(clean_texts)
        text_lower = text.lower()

        sensitive_data = []
        document_signals = []

        # ---------- NLP ENTITIES ----------
        doc = nlp(text)
        for ent in doc.ents:
            # PERSON / GPE → only if meaningful
            if ent.label_ in ["PERSON", "GPE"] and not is_ocr_noise(ent.text):
                sensitive_data.append((ent.text, ent.label_))

            # ORG → STRICT filter
            elif (
                ent.label_ == "ORG"
                and len(ent.text) >= 8
                and not is_ocr_noise(ent.text)
                and len(ent.text.split()) >= 2
            ):
                sensitive_data.append((ent.text, ent.label_))

        # ---------- REGEX PII (ALWAYS TRUSTED) ----------
        for label, pattern in PATTERNS.items():
            for match in re.findall(pattern, text):
                sensitive_data.append((match, label))

        # ---------- DOCUMENT KEYWORD DETECTION ----------
        for doc_type, keywords in DOCUMENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    document_signals.append({
                        "document_type": doc_type,
                        "evidence": kw,
                        "source": "ocr_text"
                    })
                    break

        logger.debug(f"OCR TEXT: {text}")
        logger.debug(f"OCR SENSITIVE DATA: {sensitive_data}")
        logger.debug(f"OCR DOCUMENT SIGNALS: {document_signals}")

        return text, sensitive_data, document_signals

    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return "", [], []
