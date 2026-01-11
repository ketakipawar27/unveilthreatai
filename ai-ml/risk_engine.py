# risk_engine.py

import logging
import re

logger = logging.getLogger(__name__)

# --------------------------------------------------
# RISK WEIGHTS
# --------------------------------------------------
WEIGHTS = {
    "face": 2,
    "multiple_faces": 3,

    "identity_document": 5,
    "financial_document": 6,

    "embedded_image": 2,
    "sensitive_text": 4,
    "external_link": 1,
}

# --------------------------------------------------
# RISK LEVEL MAPPING
# --------------------------------------------------
def risk_level(score: int) -> str:
    if score >= 10:
        return "CRITICAL"
    if score >= 7:
        return "HIGH"
    if score >= 4:
        return "MODERATE"
    return "LOW"


# --------------------------------------------------
# IMAGE RISK ANALYSIS
# --------------------------------------------------
def assess_image_risk(image_result: dict) -> dict:
    """
    Risk analysis for a single image (standalone or embedded)
    """
    score = 0
    reasons = []

    faces = image_result.get("face_count", 0)
    documents = image_result.get("documents", [])
    ocr_sensitive = image_result.get("ocr_sensitive", [])
    links = image_result.get("links", [])

    # ---------- Faces ----------
    if faces > 0:
        score += WEIGHTS["face"]
        reasons.append("Human face detected")

    if faces > 3:
        score += WEIGHTS["multiple_faces"]
        reasons.append("Multiple people detected")

    # ---------- Document inference ----------
    for doc in documents:
        tier = doc.get("tier", "")
        name = doc.get("document", "Unknown document")

        if tier == "HIGH":
            score += WEIGHTS["identity_document"]
            reasons.append(f"Sensitive document detected: {name}")

        elif tier == "MODERATE":
            score += 2
            reasons.append(f"Moderate-risk document detected: {name}")

    # ---------- OCR Sensitive ----------
    if ocr_sensitive:
        score += WEIGHTS["sensitive_text"]
        reasons.append("Sensitive text detected via OCR")

    # ---------- External Links ----------
    if links:
        score += WEIGHTS["external_link"]
        reasons.append("External links detected")

    return {
        "score": score,
        "level": risk_level(score),
        "reasons": reasons
    }


# --------------------------------------------------
# TEXT-BASED SENSITIVE DATA DETECTION
# --------------------------------------------------
def detect_sensitive_text(text_blocks):
    """
    Regex-based high-confidence sensitive data detection
    """
    findings = []

    patterns = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"\b(\+91|0)?[6-9]\d{9}\b",
        "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "URL": r"https?://\S+"
    }

    for block in text_blocks:
        text = block.get("text", "")
        if not text:
            continue

        for label, pattern in patterns.items():
            for match in re.findall(pattern, text):
                findings.append({
                    "type": label,
                    "value": match
                })

    return findings


# --------------------------------------------------
# DOCUMENT RISK ANALYSIS
# --------------------------------------------------
def assess_document_risk(document_result: dict) -> dict:
    """
    Document-level risk assessment
    """
    score = 0
    reasons = []

    text_blocks = document_result.get("text_blocks", [])
    links = document_result.get("links", [])
    embedded_images = document_result.get("embedded_images_count", 0)
    image_analysis = document_result.get("image_analysis", [])
    ocr_sensitive = document_result.get("ocr_sensitive", [])
    ocr_document_signals = document_result.get("ocr_document_signals", [])

    # ---------- Embedded Images ----------
    if embedded_images > 0:
        score += WEIGHTS["embedded_image"]
        reasons.append("Embedded images detected in document")

    # ---------- External Links ----------
    if links:
        score += WEIGHTS["external_link"]
        reasons.append("External links found in document")

    # ---------- Text-based Sensitive Detection ----------
    text_findings = detect_sensitive_text(text_blocks)
    if text_findings:
        score += WEIGHTS["sensitive_text"]
        reasons.append("Sensitive information found in document text")

    # ---------- OCR-based Sensitive Detection ----------
    if ocr_sensitive:
        score += WEIGHTS["sensitive_text"]
        reasons.append("Sensitive information detected in embedded images")

    # ---------- OCR Document Signals ----------
    for sig in ocr_document_signals:
        doc_type = sig.get("document_type")
        if doc_type:
            score += WEIGHTS["identity_document"]
            reasons.append(f"Sensitive document detected: {doc_type}")

    # --------------------------------------------------
    # EMBEDDED IMAGE FULL RISK MERGE
    # --------------------------------------------------
    for img in image_analysis:
        img_risk = img.get("risk_assessment", {})

        img_score = img_risk.get("score", 0)
        img_level = img_risk.get("risk_level")
        img_reasons = img_risk.get("risk_reasons", [])

        if img_score > 0:
            score += img_score
            reasons.extend(img_reasons)

        if img.get("face_count", 0) > 0:
            reasons.append("Human face detected")

        for doc in img.get("documents", []):
            reasons.append(f"Sensitive document detected: {doc.get('document')}")

        if img.get("qr_bboxes"):
            reasons.append("QR code detected in embedded image")


    return {
        "score": score,
        "level": risk_level(score),
        "reasons": list(set(reasons))
    }
