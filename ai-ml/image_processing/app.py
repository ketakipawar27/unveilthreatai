# app.py
from flask import Flask, render_template, request, jsonify
import os
import logging
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from werkzeug.utils import secure_filename
from datetime import datetime
import re
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError as e:
    MTCNN_AVAILABLE = False
    logging.error(f"MTCNN import failed: {str(e)}")
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch
from ultralytics import YOLO
import spacy
import easyocr
from deepface import DeepFace
import pkg_resources
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# Import from image_processing
from image_processing import (
    detect_faces, scene_understanding, geo_inference, brand_inference, document_inference,
    detect_qr_barcode, extract_ocr_text, extract_exif_data, reverse_image_search,
    privacy_zone_detection, analyze_caption, assess_risk, allowed_file
)

# Initialize Flask app
app = Flask(__name__)

# Configuration settings
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff'}
if HEIC_SUPPORT:
    ALLOWED_EXTENSIONS.add('.heic')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit
DEBUG_MODE = True  # Set to False in production
HAAR_CASCADE_PATH = os.path.join('static', 'haarcascade_frontalface_default.xml')
DEEPFACE_WEIGHTS_DIR = os.getenv("DEEPFACE_WEIGHTS_DIR", os.path.expanduser("~/.deepface/weights"))
AGE_MODEL_PATH = os.path.join(DEEPFACE_WEIGHTS_DIR, "age_model_weights.h5")
GENDER_MODEL_PATH = os.path.join(DEEPFACE_WEIGHTS_DIR, "gender_model_weights.h5")
EMOTION_MODEL_PATH = os.path.join(DEEPFACE_WEIGHTS_DIR, "facial_expression_model_weights.h5")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('digital_domino_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for models - initialized in image_processing, but if needed here, but since functions use them, assume they are global in import

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html', heic_support=HEIC_SUPPORT)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and perform all analyses."""
    if 'file' not in request.files:
        logger.warning("No file provided in upload request")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    caption = request.form.get('caption', '')
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            if not os.path.exists(filepath):
                logger.error(f"Failed to save file: {filepath}")
                return jsonify({"error": "Failed to save file"}), 500
            logger.debug(f"File saved: {filepath}")
            
            exif_result = extract_exif_data(filepath)
            face_count, face_confidences, face_bboxes, demographics, face_qualities = detect_faces(filepath)
            scene_description, scene_objects, object_bboxes = scene_understanding(filepath)
            geo_locations = geo_inference(filepath)
            brands, brand_bboxes = brand_inference(filepath)
            documents = document_inference(filepath)
            qr_bboxes, barcode_bboxes = detect_qr_barcode(filepath)
            ocr_text, ocr_sensitive_data = extract_ocr_text(filepath)
            reverse_search = reverse_image_search(filepath)
            caption_tone, caption_intent, caption_risks = analyze_caption(caption)
            privacy_zones = privacy_zone_detection(scene_objects, ocr_sensitive_data, filepath, qr_bboxes, barcode_bboxes, documents, brand_bboxes)
            risk_assessment = assess_risk(
                exif_result, face_count, demographics, face_qualities, scene_description, scene_objects,
                geo_locations, ocr_sensitive_data, reverse_search, caption_risks, privacy_zones, brands, documents, qr_bboxes, barcode_bboxes
            )
            
            bboxes = []
            for i, bbox in enumerate(face_bboxes):
                bboxes.append({"type": "face", "bbox": bbox, "label": f"Face {i+1}: {demographics[i]['age']}y, {demographics[i]['gender']}"})
            for i, bbox in enumerate(object_bboxes):
                if i < len(scene_objects):
                    bboxes.append({"type": "object", "bbox": bbox, "label": f"Object: {scene_objects[i][0]}"})
            for bbox in brand_bboxes:
                bboxes.append({"type": bbox["type"], "bbox": bbox["bbox"], "label": bbox["label"]})
            for zone in privacy_zones:
                if 'bbox' in zone:
                    bboxes.append({"type": zone['type'], "bbox": zone['bbox'], "label": zone['label']})
            
            result = {
                "exif": exif_result,
                "face_count": face_count,
                "face_confidences": face_confidences,
                "demographics": demographics,
                "face_qualities": face_qualities,
                "scene_description": scene_description,
                "scene_objects": scene_objects,
                "geo_locations": geo_locations,
                "brands": brands,
                "brand_bboxes": brand_bboxes,
                "documents": documents,
                "qr_bboxes": qr_bboxes,
                "barcode_bboxes": barcode_bboxes,
                "ocr_text": ocr_text,
                "ocr_sensitive_data": ocr_sensitive_data,
                "reverse_search": reverse_search,
                "caption_tone": caption_tone,
                "caption_intent": caption_intent,
                "caption_risks": caption_risks,
                "risk_assessment": risk_assessment,
                "bboxes": bboxes
            }
            
            if exif_result["status"] == "error" and DEBUG_MODE:
                result["debug_info"] = exif_result["error"]
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Error handling upload for {filename}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Server error: {str(e)}"}), 500
        
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"File deleted: {filepath}")
            except Exception as e:
                logger.error(f"Error deleting file {filepath}: {str(e)}")
    
    logger.warning(f"Unsupported file format: {file.filename}")
    return jsonify({"error": "Unsupported file format"}), 400

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)