# privacy_utils.py

import logging

logger = logging.getLogger(__name__)

def privacy_zone_detection(
    scene_objects,
    ocr_sensitive_data,
    image_path,
    qr_bboxes,
    barcode_bboxes,
    detected_documents,
    brand_bboxes
):
    """
    Create privacy zones for UI highlighting.
    Works with structured document inference output.
    """

    zones = []

    # ===================== DOCUMENT ZONES =====================
    # detected_documents is now a LIST OF DICTS (not tuples)
    for doc in detected_documents or []:
        try:
            zones.append({
                "type": "document",
                "label": f"Document: {doc.get('document', 'Unknown')}",
                "details": doc.get(
                    "risk_reason",
                    "Sensitive personal or financial document detected"
                )
                # bbox intentionally omitted (no reliable localization yet)
            })
        except Exception as e:
            logger.debug(f"Document zone error: {e}")

    # ===================== OCR TEXT ZONES =====================
    # ocr_sensitive_data = [(text, label), ...]
    for item in ocr_sensitive_data or []:
        try:
            text, label = item
            zones.append({
                "type": "text",
                "label": f"Sensitive Text ({label})",
                "details": text
            })
        except Exception as e:
            logger.debug(f"OCR zone error: {e}")

    # ===================== QR CODES =====================
    for bbox in qr_bboxes or []:
        zones.append({
            "type": "qr",
            "label": "QR Code",
            "bbox": bbox,
            "details": "Scannable data detected"
        })

    # ===================== BARCODES =====================
    for bbox in barcode_bboxes or []:
        zones.append({
            "type": "barcode",
            "label": "Barcode",
            "bbox": bbox,
            "details": "Scannable code detected"
        })

    # ===================== BRAND ZONES =====================
    # brand_bboxes = [{label, bbox, confidence, source}]
    for brand in brand_bboxes or []:
        zones.append({
            "type": "brand",
            "label": brand.get("label", "Brand"),
            "bbox": brand.get("bbox"),
            "details": "Brand exposure (profiling / ad targeting risk)"
        })

    return zones
