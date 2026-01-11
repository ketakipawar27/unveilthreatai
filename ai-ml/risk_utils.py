# risk_utils.py

import logging

logger = logging.getLogger(__name__)

RISK_PRIORITY = {"Low": 0, "Moderate": 1, "High": 2}


def escalate(current, new):
    return new if RISK_PRIORITY[new] > RISK_PRIORITY[current] else current


# --------------------------------------------------
# MAIN RISK ENGINE
# --------------------------------------------------
def assess_risk(
    exif_data,
    face_count,
    demographics,
    face_roles,
    scene_description,
    scene_objects,
    geo_locations,
    ocr_sensitive_data,
    ocr_document_signals,
    reverse_search,
    caption_risks,
    privacy_zones,
    brands,
    documents,
    qr_bboxes,
    barcode_bboxes
):
    """
    Conservative, privacy-first risk assessment.
    Uses multi-signal fusion for accuracy.
    """

    risk_level = "Low"
    risk_reasons = []
    recommendations = []
    markers = []

    # =====================================================
    # 1️⃣ EXIF METADATA
    # =====================================================
    if exif_data.get("status") == "success":
        meta = exif_data.get("data", {})

        if "google_maps_url" in meta:
            risk_level = "High"
            risk_reasons.append("Precise GPS location found in image metadata.")
            recommendations.append("Remove EXIF metadata before sharing.")

        elif meta.get("exif"):
            risk_level = escalate(risk_level, "Moderate")
            risk_reasons.append("Image contains device or timestamp metadata.")
            recommendations.append("Strip EXIF metadata to reduce tracking.")

    # =====================================================
    # 2️⃣ FACE & IDENTITY
    # =====================================================
    if face_count > 0:
        if "Primary" in face_roles:
            risk_level = "High"
            risk_reasons.append("Clear identifiable face detected.")
            recommendations.append("Blur or crop faces before sharing.")
        elif "Secondary" in face_roles:
            risk_level = escalate(risk_level, "Moderate")
            risk_reasons.append("Identifiable people visible in the image.")
            recommendations.append("Review background faces for privacy.")

        for i, demo in enumerate(demographics):
            try:
                if int(demo.get("age", 99)) < 18:
                    risk_level = "High"
                    risk_reasons.append("Minor detected — child privacy risk.")
                    recommendations.append("Avoid sharing images containing minors.")
            except Exception:
                pass

        for i, role in enumerate(face_roles):
            markers.append({
                "type": "face",
                "label": f"Face {i+1}",
                "details": role
            })

    # =====================================================
    # 3️⃣ SCENE CONTEXT
    # =====================================================
    PRIVATE_SCENES = {"bedroom", "home", "office", "kids", "living room"}

    if scene_description:
        if any(k in scene_description.lower() for k in PRIVATE_SCENES):
            risk_level = escalate(risk_level, "Moderate")
            risk_reasons.append("Image appears to be taken in a private space.")
            recommendations.append("Avoid sharing images from private environments.")

    # =====================================================
    # 4️⃣ DOCUMENT FUSION (OCR + VISION)
    # =====================================================
    ocr_docs = {d["document_type"] for d in (ocr_document_signals or [])}
    visual_docs = {d.get("document") for d in (documents or []) if isinstance(d, dict)}

    if ocr_docs or visual_docs:
        risk_level = escalate(risk_level, "Moderate")
        risk_reasons.append("Possible personal or financial document detected.")
        recommendations.append("Review image before sharing.")

    if ocr_docs & visual_docs:
        risk_level = "High"
        risk_reasons.append("OCR and visual analysis both confirm a sensitive document.")
        recommendations.append("Do not share images containing personal documents.")

    for doc in sorted(ocr_docs | visual_docs):
        markers.append({
            "type": "document",
            "label": "Sensitive Document",
            "details": doc
        })

    # =====================================================
    # 5️⃣ LOCATION INFERENCE
    # =====================================================
    for geo in geo_locations or []:
        if geo.get("status") == "confirmed":
            risk_level = "High"
            risk_reasons.append(f"Exact location identified: {geo.get('location')}.")
            recommendations.append("Avoid sharing images with location metadata.")

    # =====================================================
    # 6️⃣ OCR TEXT (STRICT ESCALATION)
    # =====================================================
    NUMERIC_PII = {"AADHAAR", "PAN", "CARD_NUMBER", "PHONE", "PINCODE"}

    for text, label in ocr_sensitive_data or []:
        if label in NUMERIC_PII:
            risk_level = "High"
        elif label in {"PERSON", "ORG"}:
            if len(text) < 8 or len(text.split()) < 2:
                continue
            risk_level = escalate(risk_level, "Moderate")
        else:
            continue

        risk_reasons.append(f"Sensitive text detected: {label}.")
        recommendations.append("Blur or redact sensitive text.")
        markers.append({
            "type": "text",
            "label": label,
            "details": text
        })

    # =====================================================
    # 7️⃣ QR / BARCODES
    # =====================================================
    if qr_bboxes or barcode_bboxes:
        risk_level = "High"
        risk_reasons.append("QR code or barcode detected.")
        recommendations.append("Blur QR codes and barcodes.")

    # =====================================================
    # 8️⃣ BRANDS & PROFILING
    # =====================================================
    if brands:
        risk_level = escalate(risk_level, "Moderate")
        risk_reasons.append("Brand logos detected — potential profiling risk.")
        recommendations.append("Blur logos if privacy is a concern.")

    # =====================================================
    # 9️⃣ FINAL MARKERS
    # =====================================================
    for zone in privacy_zones or []:
        markers.append(zone)

    if risk_level == "Low":
        recommendations.append("No major privacy risks detected.")

    return {
        "risk_level": risk_level,
        "risk_reasons": list(dict.fromkeys(risk_reasons)),
        "recommendations": list(dict.fromkeys(recommendations)),
        "markers": markers
    }
