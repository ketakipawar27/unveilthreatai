import logging

from exif_utils import extract_exif_data
from face_utils import detect_faces
from scene_utils import scene_understanding
from geo_utils import geo_inference
from brand_utils import brand_inference
from document_utils import document_inference
from qr_utils import detect_qr_barcode
from ocr_utils import extract_ocr_text
from reverse_utils import reverse_image_search
from privacy_utils import privacy_zone_detection
from caption_utils import analyze_caption
from risk_utils import assess_risk

logger = logging.getLogger(__name__)


# --------------------------------------------------
# MAIN IMAGE PIPELINE
# --------------------------------------------------
def process_image(image_path, caption=""):
    """
    Unified image analysis pipeline.
    Used by:
    - app.py (image uploads)
    - document_pipeline.py (embedded images)
    """

    # ---------- EXIF ----------
    exif_result = extract_exif_data(image_path)

    # ---------- FACE ----------
    (
        face_count,
        face_confidences,
        face_bboxes,
        demographics,
        face_roles
    ) = detect_faces(image_path)

    # ---------- SCENE ----------
    scene_description, scene_objects, object_bboxes = scene_understanding(image_path)

    # ---------- GEO ----------
    geo_locations = geo_inference(image_path)

    # ---------- BRAND ----------
    brands, brand_bboxes = brand_inference(image_path)

    # ---------- DOCUMENT IN IMAGE ----------
    documents = document_inference(image_path)

    # ---------- QR / BARCODE ----------
    qr_bboxes, barcode_bboxes = detect_qr_barcode(image_path)

    # ---------- OCR ----------
    ocr_text, ocr_sensitive_data, ocr_document_signals = extract_ocr_text(image_path)

    # ---------- REVERSE SEARCH ----------
    reverse_search = reverse_image_search(image_path)

    # ---------- CAPTION ----------
    caption_tone, caption_intent, caption_risks = analyze_caption(caption)

    # ---------- PRIVACY ZONES ----------
    privacy_zones = privacy_zone_detection(
        scene_objects,
        ocr_sensitive_data,
        image_path,
        qr_bboxes,
        barcode_bboxes,
        documents,
        brand_bboxes
    )

    # ---------- RISK ----------
    risk_assessment = assess_risk(
        exif_result,
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
    )

    return {
        "exif": exif_result,
        "face_count": face_count,
        "face_confidences": face_confidences,
        "face_bboxes": face_bboxes,
        "demographics": demographics,
        "face_roles": face_roles,

        "scene_description": scene_description,
        "scene_objects": scene_objects,
        "object_bboxes": object_bboxes,

        "geo_locations": geo_locations,
        "brands": brands,
        "brand_bboxes": brand_bboxes,
        "documents": documents,

        "qr_bboxes": qr_bboxes,
        "barcode_bboxes": barcode_bboxes,

        "ocr_text": ocr_text,
        "ocr_sensitive_data": ocr_sensitive_data,
        "ocr_document_signals": ocr_document_signals,

        "reverse_search": reverse_search,

        "caption_tone": caption_tone,
        "caption_intent": caption_intent,
        "caption_risks": caption_risks,

        "privacy_zones": privacy_zones,
        "risk_assessment": risk_assessment
    }
