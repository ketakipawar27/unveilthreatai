# document_analyzer.py

import os
import logging
import tempfile

from document_loader import detect_document_type
from document_text_extractor import extract_document_text
from document_link_extractor import extract_links
from document_image_extractor import (
    extract_images_from_pdf,
    extract_images_from_docx,
    extract_images_from_pptx
)

from image_pipeline import analyze_image  # ✅ existing image pipeline

logger = logging.getLogger(__name__)

# --------------------------------------------------
# MASTER DOCUMENT ANALYZER
# --------------------------------------------------
def analyze_document(file_path):
    """
    Extend image analysis to documents:
    - extract text
    - extract links
    - extract embedded images
    - run embedded images through image_pipeline
    """

    result = {
        "type": "document",
        "file": os.path.basename(file_path),
        "document_type": None,
        "text_blocks": [],
        "links": [],
        "embedded_images_count": 0,
        "image_analysis": []
    }

    # --------------------------------------------------
    # DETECT DOCUMENT TYPE
    # --------------------------------------------------
    doc_info = detect_document_type(file_path)

    if not doc_info["valid"]:
        result["error"] = doc_info["reason"]
        return result

    doc_type = doc_info["doc_type"]
    result["document_type"] = doc_type

    # --------------------------------------------------
    # TEXT EXTRACTION
    # --------------------------------------------------
    try:
        result["text_blocks"] = extract_document_text(file_path, doc_type)
    except Exception as e:
        logger.error(f"Text extraction failed: {e}", exc_info=True)

    # --------------------------------------------------
    # LINK EXTRACTION
    # --------------------------------------------------
    try:
        result["links"] = extract_links(file_path)
    except Exception as e:
        logger.error(f"Link extraction failed: {e}", exc_info=True)

    # --------------------------------------------------
    # EMBEDDED IMAGE EXTRACTION
    # --------------------------------------------------
    try:
        if doc_type == "pdf":
            embedded = extract_images_from_pdf(file_path)
        elif doc_type == "docx":
            embedded = extract_images_from_docx(file_path)
        elif doc_type == "pptx":
            embedded = extract_images_from_pptx(file_path)
        else:
            embedded = []
    except Exception as e:
        logger.error(f"Embedded image extraction failed: {e}", exc_info=True)
        embedded = []

    result["embedded_images_count"] = len(embedded)

    # --------------------------------------------------
    # IMAGE ANALYSIS (REUSE IMAGE PIPELINE)
    # --------------------------------------------------
    for idx, img in enumerate(embedded):
        img_path = img.get("path")

        if not img_path or not os.path.exists(img_path):
            continue

        try:
            analysis = analyze_image(img_path)

            analysis["embedded_index"] = idx + 1
            analysis["source"] = "embedded_document_image"

            result["image_analysis"].append(analysis)

        except Exception as e:
            logger.warning(f"Embedded image analysis failed: {e}")

    return result
