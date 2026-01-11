# document_pipeline.py

import logging
import os

from document_loader import detect_document_type
from document_text_extractor import extract_document_text
from document_image_extractor import (
    extract_images_from_pdf,
    extract_images_from_docx,
    extract_images_from_pptx
)
from document_link_extractor import extract_links
from risk_engine import assess_document_risk, detect_sensitive_text

# 🔥 REUSE IMAGE PIPELINE
from image_pipeline import process_image

logger = logging.getLogger(__name__)

# --------------------------------------------------
# MAIN DOCUMENT PIPELINE
# --------------------------------------------------
def process_document(file_path):
    """
    Full document processing pipeline
    Includes FULL embedded image risk analysis
    """

    # ---------- detect document type ----------
    doc_info = detect_document_type(file_path)

    if not doc_info["valid"]:
        logger.warning(f"Document rejected: {doc_info['reason']}")
        return {
            "type": "document",
            "risk": {
                "level": "UNKNOWN",
                "score": 0,
                "reasons": [doc_info["reason"]]
            },
            "details": {
                "error": doc_info["reason"]
            }
        }

    doc_type = doc_info["doc_type"]

    # ---------- extract text ----------
    try:
        text_blocks = extract_document_text(file_path, doc_type)
    except Exception as e:
        logger.error(f"Text extraction failed: {e}", exc_info=True)
        text_blocks = []

    # ---------- detect sensitive text ----------
    try:
        sensitive_data = detect_sensitive_text(text_blocks)
    except Exception as e:
        logger.error(f"Sensitive text detection failed: {e}", exc_info=True)
        sensitive_data = []

    # ---------- extract embedded images ----------
    embedded_images = []

    try:
        if doc_type == "pdf":
            embedded_images = extract_images_from_pdf(file_path)
        elif doc_type == "docx":
            embedded_images = extract_images_from_docx(file_path)
        elif doc_type == "pptx":
            embedded_images = extract_images_from_pptx(file_path)
    except Exception as e:
        logger.error(f"Embedded image extraction failed: {e}", exc_info=True)
        embedded_images = []

    # --------------------------------------------------
    # 🔥 FULL IMAGE ANALYSIS ON EMBEDDED IMAGES
    # --------------------------------------------------
    image_analysis = []

    for img in embedded_images:
        img_path = img.get("path") or img.get("image_path")

        if not img_path or not os.path.exists(img_path):
            continue

        try:
            logger.info(f"Analyzing embedded image: {img_path}")

            img_result = process_image(img_path, caption="")

            # ✅ NORMALIZED STRUCTURE FOR RISK ENGINE
            image_analysis.append({
                "image_path": img_path,
                "risk": img_result.get("risk_assessment", {}),
                "documents": img_result.get("documents", []),
                "ocr_sensitive": img_result.get("ocr_sensitive_data", []),
                "face_count": img_result.get("face_count", 0),
                "qr_detected": bool(img_result.get("qr_bboxes")),
            })

        except Exception as e:
            logger.error(
                f"Embedded image analysis failed for {img_path}: {e}",
                exc_info=True
            )
            logger.debug(f"Embedded image risk: {img_result['risk_assessment']}")


    # ---------- extract links ----------
    try:
        links = extract_links(file_path)
    except Exception as e:
        logger.error(f"Link extraction failed: {e}", exc_info=True)
        links = []

    # ---------- assemble result ----------
    result = {
        "document_type": doc_type.upper(),
        "pages": len(text_blocks),

        "text_blocks": text_blocks,
        "sensitive_data": sensitive_data,
        "links": links,

        "embedded_images_count": len(embedded_images),
        "embedded_image_objects": embedded_images,

        # 🔥 USED BY RISK ENGINE
        "image_analysis": image_analysis
    }

    # ---------- assess risk ----------
    risk = assess_document_risk(result)

    return {
        "type": "document",
        "risk": risk,
        "details": result
    }
