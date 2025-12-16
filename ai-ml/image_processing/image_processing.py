# image_processing.py
import os
import logging
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import re
from geopy.geocoders import Nominatim
import spacy
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError as e:
    MTCNN_AVAILABLE = False
    logging.error(f"MTCNN import failed: {str(e)}")
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch
from ultralytics import YOLO
import easyocr
from deepface import DeepFace
import pkg_resources

logger = logging.getLogger(__name__)

# Check DeepFace weights directory
DEEPFACE_WEIGHTS_DIR = os.getenv("DEEPFACE_WEIGHTS_DIR", os.path.expanduser("~/.deepface/weights"))
AGE_MODEL_PATH = os.path.join(DEEPFACE_WEIGHTS_DIR, "age_model_weights.h5")
GENDER_MODEL_PATH = os.path.join(DEEPFACE_WEIGHTS_DIR, "gender_model_weights.h5")
EMOTION_MODEL_PATH = os.path.join(DEEPFACE_WEIGHTS_DIR, "facial_expression_model_weights.h5")

# Check DeepFace version
try:
    deepface_version = pkg_resources.get_distribution("deepface").version
    logger.info(f"DeepFace version: {deepface_version}")
    if deepface_version < "0.0.79":
        logger.warning("DeepFace version is older than 0.0.79. Consider upgrading: pip install deepface --upgrade")
except pkg_resources.DistributionNotFound:
    logger.error("DeepFace package not found. Please install it: pip install deepface")
    raise

# Load models
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    yolo_model = YOLO("yolov8n.pt")
    nlp = spacy.load("en_core_web_sm")
    if MTCNN_AVAILABLE:
        face_detector = MTCNN()
        logger.info("MTCNN face detector loaded successfully.")
    HAAR_CASCADE_PATH = os.path.join('static', 'haarcascade_frontalface_default.xml')
    if os.path.exists(HAAR_CASCADE_PATH):
        haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        logger.info(f"Haar Cascade loaded from {HAAR_CASCADE_PATH}")
    else:
        logger.warning(f"Haar Cascade file not found at {HAAR_CASCADE_PATH}. Fallback face detection disabled.")
        haar_cascade = None
    # Preload DeepFace models
    deepface_models_loaded = {'Age': False, 'Gender': False, 'Emotion': False}
    if os.path.exists(AGE_MODEL_PATH):
        try:
            DeepFace.build_model('Age')
            deepface_models_loaded['Age'] = True
            logger.info(f"DeepFace age model preloaded successfully from {AGE_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load DeepFace age model: {str(e)}. Age analysis will be skipped.")
    else:
        logger.warning(f"Age model weights not found at {AGE_MODEL_PATH}. Age analysis will be skipped.")
    if os.path.exists(GENDER_MODEL_PATH):
        try:
            DeepFace.build_model('Gender')
            deepface_models_loaded['Gender'] = True
            logger.info(f"DeepFace gender model preloaded successfully from {GENDER_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load DeepFace gender model: {str(e)}. Gender analysis will be skipped.")
    else:
        logger.warning(f"Gender model weights not found at {GENDER_MODEL_PATH}. Gender analysis will be skipped.")
    if os.path.exists(EMOTION_MODEL_PATH):
        try:
            DeepFace.build_model('Emotion')
            deepface_models_loaded['Emotion'] = True
            logger.info(f"DeepFace emotion model preloaded successfully from {EMOTION_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load DeepFace emotion model: {str(e)}. Emotion analysis will be skipped.")
    else:
        logger.warning(f"Emotion model weights not found at {EMOTION_MODEL_PATH}. Emotion analysis will be skipped.")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}", exc_info=True)
    raise

# Initialize EasyOCR
try:
    ocr_reader = easyocr.Reader(['en'], gpu=False)  # Use GPU if available in production
    logger.info("EasyOCR initialized successfully for English")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {str(e)}")
    ocr_reader = None

# Check DeepFace weights directory
if not os.path.exists(DEEPFACE_WEIGHTS_DIR):
    logger.warning(f"DeepFace weights directory not found at {DEEPFACE_WEIGHTS_DIR}. Please ensure weights are downloaded.")

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.heic'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def convert_decimal_degrees(degree, minutes, seconds, direction):
    """Convert GPS coordinates from degrees, minutes, seconds to decimal degrees."""
    try:
        decimal_degrees = degree + minutes / 60 + seconds / 3600
        if direction in ("S", "W"):
            decimal_degrees *= -1
        return decimal_degrees
    except (TypeError, ValueError) as e:
        logger.debug(f"Error converting coordinates: {e}")
        return None

def create_google_maps_url(gps_coords):
    """Create a Google Maps URL from GPS coordinates."""
    try:
        dec_deg_lat = convert_decimal_degrees(
            float(gps_coords["lat"][0]),
            float(gps_coords["lat"][1]),
            float(gps_coords["lat"][2]),
            gps_coords["lat_ref"]
        )
        dec_deg_lon = convert_decimal_degrees(
            float(gps_coords["lon"][0]),
            float(gps_coords["lon"][1]),
            float(gps_coords["lon"][2]),
            gps_coords["lon_ref"]
        )
        if dec_deg_lat is None or dec_deg_lon is None:
            return None
        return f"https://maps.google.com/?q={dec_deg_lat:.6f},{dec_deg_lon:.6f}"
    except (KeyError, ValueError) as e:
        logger.debug(f"Error creating Google Maps URL: {e}")
        return None

def detect_faces(image_path):
    """Detect faces in the image using MTCNN or Haar Cascade, and analyze demographics using DeepFace. Also assess face quality for deepfake risk."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image for face detection: {image_path}")
            return 0, [], [], [], []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_count = 0
        confidences = []
        bboxes = []
        demographics = []
        face_qualities = []

        # Try MTCNN first
        if MTCNN_AVAILABLE:
            try:
                faces = face_detector.detect_faces(img_rgb)
                face_count = len(faces)
                confidences = [face['confidence'] for face in faces]
                bboxes = [face['box'] for face in faces]  # [x, y, width, height]
                logger.debug(f"MTCNN detected {face_count} faces with confidences: {confidences}")
            except Exception as e:
                logger.error(f"MTCNN face detection failed: {str(e)}", exc_info=True)
                face_count = 0
                confidences = []
                bboxes = []

        # Fallback to Haar Cascade if MTCNN fails or is unavailable
        if face_count == 0 and haar_cascade is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_count = len(faces)
                confidences = [1.0] * face_count  # Haar doesn't provide confidence
                bboxes = [[x, y, w, h] for (x, y, w, h) in faces]
                logger.debug(f"Haar Cascade detected {face_count} faces")
            except Exception as e:
                logger.error(f"Haar Cascade face detection failed: {str(e)}", exc_info=True)

        # Check available DeepFace models
        available_actions = [action.lower() for action, loaded in deepface_models_loaded.items() if loaded]

        # Demographic analysis and quality assessment
        for i in range(face_count):
            try:
                x, y, w, h = bboxes[i]
                face_img = img_rgb[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                # Simple sharpness measure for deepfake risk
                sharpness = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                quality = 'High' if sharpness > 100 else 'Low'  # Threshold can be adjusted
                face_qualities.append(quality)
                if available_actions:
                    analysis = DeepFace.analyze(face_img, actions=available_actions, enforce_detection=False)
                    demo = {
                        'age': analysis[0].get('age', 'Unknown') if 'age' in available_actions else 'Unknown',
                        'gender': analysis[0].get('dominant_gender', 'Unknown') if 'gender' in available_actions else 'Unknown',
                        'emotion': analysis[0].get('dominant_emotion', 'Unknown') if 'emotion' in available_actions else 'Unknown'
                    }
                else:
                    demo = {'age': 'Unknown', 'gender': 'Unknown', 'emotion': 'Unknown'}
                    logger.debug(f"No DeepFace models available for face {i+1} analysis")
                demographics.append(demo)
            except Exception as e:
                logger.debug(f"Demographic analysis failed for face {i+1}: {str(e)}")
                demographics.append({'age': 'Unknown', 'gender': 'Unknown', 'emotion': 'Unknown'})
                face_qualities.append('Unknown')

        logger.debug(f"Final face detection: {face_count} faces, confidences: {confidences}")
        return face_count, confidences, bboxes, demographics, face_qualities

    except Exception as e:
        logger.error(f"Error in face detection for {image_path}: {str(e)}", exc_info=True)
        return 0, [], [], [], []

def scene_understanding(image_path):
    """Generate scene description using BLIP and detect objects using YOLO, including expanded classes for luxury, workplace, etc."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return "", [], []
        img = Image.open(image_path).convert('RGB')
        inputs = blip_processor(images=img, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        scene_description = blip_processor.decode(outputs[0], skip_special_tokens=True)
        
        yolo_results = yolo_model(image_path)
        objects = []
        bboxes = []
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls)
                label = result.names[cls]
                conf = box.conf.item()
                conf = float(conf)
                if conf > 0.5 and label in ['person', 'laptop', 'book', 'cell phone', 'car', 'bottle', 'key', 'handbag', 'tie', 'wine glass', 'suitcase', 'keyboard', 'mouse', 'tv']:
                    objects.append((label, conf))
                    bboxes.append(box.xywh.tolist()[0])  # [x_center, y_center, width, height]
        
        logger.debug(f"Scene: {scene_description}, Objects: {objects}")
        return scene_description, objects, bboxes
    except Exception as e:
        logger.error(f"Error in scene understanding for {image_path}: {str(e)}", exc_info=True)
        return "", [], []

def geo_inference(image_path):
    """Infer geographical locations using EXIF GPS data with reverse geocoding, enhanced NLP for OCR text, and CLIP as fallback, with confidence >= 0.7."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return []
        
        inferred_locations = []
        seen_locations = set()

        # 1. EXIF GPS-based location detection
        exif_data = extract_exif_data(image_path)
        if exif_data["status"] == "success" and "google_maps_url" in exif_data["data"]:
            try:
                geolocator = Nominatim(user_agent="digital_domino_analyzer")
                gps_coords = exif_data["data"]["exif"]
                lat = convert_decimal_degrees(
                    float(gps_coords["GPSLatitude"][0]),
                    float(gps_coords["GPSLatitude"][1]),
                    float(gps_coords["GPSLatitude"][2]),
                    gps_coords["GPSLatitudeRef"]
                )
                lon = convert_decimal_degrees(
                    float(gps_coords["GPSLongitude"][0]),
                    float(gps_coords["GPSLongitude"][1]),
                    float(gps_coords["GPSLongitude"][2]),
                    gps_coords["GPSLongitudeRef"]
                )
                if lat is not None and lon is not None:
                    location = geolocator.reverse((lat, lon), language="en")
                    if location and location.address:
                        address = location.address
                        inferred_locations.append(f"{address} (Confidence: 0.95)")  # High confidence for GPS
                        seen_locations.add(address.split(",")[0])  # Use primary location name
                        logger.debug(f"GPS location: {address}")
            except Exception as e:
                logger.debug(f"Reverse geocoding failed: {str(e)}")

        # 2. OCR-based location detection with enhanced NLP
        ocr_text, _ = extract_ocr_text(image_path)
        try:
            nlp_lg = spacy.load("en_core_web_lg")  # Use larger model for better GPE detection
            doc = nlp_lg(ocr_text)
            locations = [(ent.text, 0.9) for ent in doc.ents if ent.label_ == 'GPE' and ent.text not in seen_locations]
            inferred_locations.extend([f"{loc[0]} (Confidence: {loc[1]:.2f})" for loc in locations])
            seen_locations.update(loc[0] for loc in locations)
            logger.debug(f"OCR locations: {locations}")
        except Exception as e:
            logger.warning(f"Failed to load en_core_web_lg or process OCR text: {str(e)}")
            # Fallback to existing nlp model
            doc = nlp(ocr_text)
            locations = [(ent.text, 0.9) for ent in doc.ents if ent.label_ == 'GPE' and ent.text not in seen_locations]
            inferred_locations.extend([f"{loc[0]} (Confidence: {loc[1]:.2f})" for loc in locations])
            seen_locations.update(loc[0] for loc in locations)

        # 3. CLIP-based landmark detection as fallback
        if not inferred_locations:
            img = Image.open(image_path).convert('RGB')
            landmarks = [
                "Eiffel Tower", "Taj Mahal", "Statue of Liberty", "Big Ben", "Sydney Opera House",
                "Colosseum", "Great Wall of China", "Machu Picchu", "Pyramids of Giza", "Golden Gate Bridge",
                "Burj Khalifa", "Sagrada Familia", "Leaning Tower of Pisa", "Mount Fuji", "Christ the Redeemer"
            ]
            inputs = clip_processor(text=landmarks, images=img, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
            detected_landmarks = [(landmark, float(prob)) for landmark, prob in zip(landmarks, probs) if prob >= 0.7]
            if not detected_landmarks:
                detected_landmarks = [(landmark, float(prob)) for landmark, prob in zip(landmarks, probs) if prob >= 0.6]
                logger.debug("No landmarks with confidence >= 0.7, lowering to 0.6")
            inferred_locations.extend([f"{landmark[0]} (Confidence: {landmark[1]:.2f})" for landmark in detected_landmarks if landmark[0] not in seen_locations])
            logger.debug(f"CLIP landmarks: {detected_landmarks}")

        # Log if no locations detected
        if not inferred_locations:
            logger.warning(f"No locations detected for {image_path}")

        logger.debug(f"Geo-inference: {inferred_locations}")
        return inferred_locations
    except Exception as e:
        logger.error(f"Error in geo-inference for {image_path}: {str(e)}", exc_info=True)
        return []

def brand_inference(image_path):
    """Infer brands using CLIP, OCR for brand text, and logo detection with YOLO, with confidence >= 0.7."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return [], []

        # Load image
        img = Image.open(image_path).convert('RGB')
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            logger.error(f"Failed to load image for brand inference: {image_path}")
            return [], []

        # Expanded brand list (can be loaded from a database or file for scalability)
        brands = [
            "Coca Cola", "Nike", "Apple", "Google", "Mercedes", "Louis Vuitton", "Rolex",
            "Adidas", "Pepsi", "BMW", "Gucci", "Prada", "Samsung", "Microsoft"
        ]

        # 1. CLIP-based brand detection
        inputs = clip_processor(text=brands, images=img, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
        clip_detected_brands = [(brand, float(prob)) for brand, prob in zip(brands, probs) if prob >= 0.7]

        # 2. OCR-based brand text detection
        ocr_detected_brands = []
        brand_bboxes = []
        if ocr_reader:
            ocr_results = ocr_reader.readtext(img_cv, detail=1)  # detail=1 returns text and bboxes
            for (bbox, text, prob) in ocr_results:
                # Match text against brand names (case-insensitive)
                for brand in brands:
                    if brand.lower() in text.lower() and prob >= 0.7:
                        ocr_detected_brands.append((brand, float(prob)))
                        # Convert bbox format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] to [x, y, w, h]
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x, y = min(x_coords), min(y_coords)
                        w, h = max(x_coords) - x, max(y_coords) - y
                        brand_bboxes.append([x, y, w, h])

        # 3. Logo detection using YOLO
        logo_detected_brands = []
        logo_bboxes = []
        yolo_results = yolo_model(image_path)
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls)
                label = result.names[cls]
                conf = box.conf.item()
                # Check if label matches any brand (requires YOLO model fine-tuned for logos)
                for brand in brands:
                    if brand.lower() in label.lower() and conf >= 0.7:
                        logo_detected_brands.append((brand, float(conf)))
                        logo_bboxes.append(box.xywh.tolist()[0])  # [x_center, y_center, width, height]

        # 4. Combine results, removing duplicates
        detected_brands = []
        seen_brands = set()
        for brand, conf in clip_detected_brands:
            if brand not in seen_brands:
                detected_brands.append((brand, conf, "CLIP"))
                seen_brands.add(brand)
        for brand, conf in ocr_detected_brands:
            if brand not in seen_brands:
                detected_brands.append((brand, conf, "OCR"))
                seen_brands.add(brand)
        for brand, conf in logo_detected_brands:
            if brand not in seen_brands:
                detected_brands.append((brand, conf, "Logo"))
                seen_brands.add(brand)

        # Combine bounding boxes (prioritize logo and OCR bboxes)
        all_bboxes = []
        for bbox in brand_bboxes:
            all_bboxes.append({"type": "brand_text", "bbox": bbox, "label": "Brand Text"})
        for bbox in logo_bboxes:
            all_bboxes.append({"type": "brand_logo", "bbox": bbox, "label": "Brand Logo"})

        logger.debug(f"Brand inference: {detected_brands}, Bounding boxes: {all_bboxes}")
        return detected_brands, all_bboxes

    except Exception as e:
        logger.error(f"Error in brand inference for {image_path}: {str(e)}", exc_info=True)
        return [], []

def document_inference(image_path):
    """Infer personal documents using CLIP matching against document types."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return []
        img = Image.open(image_path).convert('RGB')
        documents = ["credit card", "debit card", "passport", "driver license", "ID card", "bank statement", "invoice", "contract", "ticket", "Aadhaar card", "PAN card"]
        inputs = clip_processor(text=documents, images=img, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
        detected_documents = [(doc, float(prob)) for doc, prob in zip(documents, probs) if prob > 0.3]
        
        logger.debug(f"Document inference: {detected_documents}")
        return detected_documents
    except Exception as e:
        logger.error(f"Error in document inference for {image_path}: {str(e)}", exc_info=True)
        return []

def detect_qr_barcode(image_path):
    """Detect QR codes using OpenCV and barcodes using gradient method."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return [], []
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image for QR/barcode detection: {image_path}")
            return [], []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # QR Detection
        qr_detector = cv2.QRCodeDetector()
        qr_data, qr_points, _ = qr_detector.detectAndDecode(gray)
        qr_bboxes = [qr_points.tolist()[0]] if qr_points is not None else []
        qr_detected = len(qr_bboxes) > 0
        
        # Barcode Detection (simple gradient method)
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        barcode_bboxes = []
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            if w / h > 5 and w > 100:  # Aspect ratio for barcodes
                barcode_bboxes.append([x, y, w, h])
        
        logger.debug(f"QR detected: {qr_detected}, Barcodes: {len(barcode_bboxes)}")
        return qr_bboxes, barcode_bboxes
    except Exception as e:
        logger.error(f"Error in QR/barcode detection for {image_path}: {str(e)}", exc_info=True)
        return [], []

def extract_ocr_text(image_path):
    """Extract text from image using EasyOCR and identify sensitive entities, including specific PII patterns."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return "", []
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image for OCR: {image_path}")
            return "", []
        if ocr_reader is None:
            logger.error("EasyOCR not initialized")
            return "", []
        # Perform OCR with EasyOCR
        results = ocr_reader.readtext(img, detail=0)  # detail=0 returns only text
        text = " ".join(results)  # Combine all detected text
        doc = nlp(text)
        sensitive_data = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'GPE', 'ORG']:
                sensitive_data.append((ent.text, ent.label_))
        for token in doc:
            if any(keyword in token.text.lower() for keyword in ['confidential', 'pan', 'account', 'ssn']):
                sensitive_data.append((token.text, 'SENSITIVE'))
        
        # Additional PII patterns
        # Aadhaar: xxxx xxxx xxxx
        aadhaar_matches = re.findall(r'\d{4}\s?\d{4}\s?\d{4}', text)
        for match in aadhaar_matches:
            sensitive_data.append((match, 'AADHAAR'))
        
        # PAN: ABCDE1234F
        pan_matches = re.findall(r'[A-Z]{5}\d{4}[A-Z]', text)
        for match in pan_matches:
            sensitive_data.append((match, 'PAN'))
        
        # Phone numbers: +91-xxxxxxxxxx, 0xxxxxxxxxx, etc.
        phone_matches = re.findall(r'(\+91|0)?[ -]?\d{10}', text)
        for match in phone_matches:
            sensitive_data.append((match[0], 'PHONE'))
        
        # Addresses: Simple, look for pincode or keywords
        address_matches = re.findall(r'\b\d{6}\b', text)  # Indian pincode
        for match in address_matches:
            sensitive_data.append((match, 'ADDRESS_PIN'))
        
        # Signatures: If "signature" in text
        if 'signature' in text.lower() or 'sign' in text.lower():
            sensitive_data.append(('Signature detected', 'SIGNATURE'))
        
        # License plates (Indian): e.g., DL01AB1234
        plate_matches = re.findall(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', text)
        for match in plate_matches:
            sensitive_data.append((match, 'LICENSE_PLATE'))
        
        logger.debug(f"OCR text: {text}, Sensitive data: {sensitive_data}")
        return text, sensitive_data
    except Exception as e:
        logger.error(f"Error in OCR for {image_path}: {str(e)}", exc_info=True)
        return "", []

def extract_exif_data(image_path):
    """Extract EXIF metadata from the image, including GPS data."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return {"status": "error", "error": "Image file not found", "data": {"filename": os.path.basename(image_path)}}
        with Image.open(image_path) as image:
            logger.debug(f"Processing image: {image_path}, format: {image.format}")
            exif_data = image._getexif()
            metadata = {
                "filename": os.path.basename(image_path),
                "format": image.format or "Unknown",
                "size": image.size,
                "mode": image.mode,
                "exif": {},
                "raw_exif": {}
            }
            
            if not exif_data:
                logger.info(f"No EXIF data found in {image_path}")
                return {"status": "no_exif", "data": metadata}
            
            gps_coords = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, f"Unknown_{tag}")
                metadata["raw_exif"][tag_name] = str(value)
                if tag_name == "GPSInfo":
                    for key, val in value.items():
                        gps_tag = GPSTAGS.get(key, f"GPS_Unknown_{key}")
                        if gps_tag == "GPSLatitude":
                            gps_coords["lat"] = val
                        elif gps_tag == "GPSLongitude":
                            gps_coords["lon"] = val
                        elif gps_tag == "GPSLatitudeRef":
                            gps_coords["lat_ref"] = val
                        elif gps_tag == "GPSLongitudeRef":
                            gps_coords["lon_ref"] = val
                        metadata["exif"][gps_tag] = str(val)
                else:
                    if tag_name == "DateTime":
                        try:
                            metadata["exif"][tag_name] = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            metadata["exif"][tag_name] = str(value)
                    else:
                        metadata["exif"][tag_name] = str(value)
            
            if gps_coords:
                metadata["google_maps_url"] = create_google_maps_url(gps_coords)
            
            return {"status": "success", "data": metadata}
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e), "data": {"filename": os.path.basename(image_path)}}

def reverse_image_search(image_path):
    """Perform a simulated reverse image search using image hash (placeholder)."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return {"found": False, "details": "Image file not found"}
        img = Image.open(image_path).convert('RGB')
        img_hash = str(hash(img.tobytes()))
        logger.debug(f"Reverse search hash: {img_hash}")
        return {"found": False, "details": "No matches found (simulated search)"}
    except Exception as e:
        logger.error(f"Error in reverse image search for {image_path}: {str(e)}", exc_info=True)
        return {"found": False, "details": str(e)}

def privacy_zone_detection(scene_objects, ocr_sensitive_data, image_path, qr_bboxes, barcode_bboxes, detected_documents, brand_bboxes):
    """Detect privacy-sensitive zones based on objects, OCR data, QR/barcodes, and documents."""
    privacy_zones = []
    for obj in scene_objects:
        if obj[0] in ['laptop', 'cell phone', 'book', 'key', 'handbag', 'tie', 'wine glass', 'suitcase', 'keyboard', 'mouse', 'tv']:
            privacy_zones.append({"type": "object", "label": f"Sensitive: {obj[0]}", "bbox": obj[2], "details": f"Confidence: {obj[1]:.2f}"})
    for data, label in ocr_sensitive_data:
        if label in ['PERSON', 'SENSITIVE', 'AADHAAR', 'PAN', 'PHONE', 'ADDRESS_PIN', 'SIGNATURE', 'LICENSE_PLATE']:
            privacy_zones.append({"type": "text", "label": f"Sensitive Text: {data}", "bbox": [50, 50, 100, 50], "details": f"Type: {label}"})

    for bbox in qr_bboxes:
        privacy_zones.append({"type": "qr", "label": "QR Code", "bbox": bbox, "details": "Potential data leak"})
    for bbox in barcode_bboxes:
        privacy_zones.append({"type": "barcode", "label": "Barcode", "bbox": bbox, "details": "Potential product/info leak"})
    
    for doc, conf in detected_documents:
        privacy_zones.append({"type": "document", "label": f"Document: {doc}", "bbox": [100, 100, 200, 200], "details": f"Confidence: {conf:.2f}"})

    for bbox in brand_bboxes:
        privacy_zones.append({
            "type": bbox["type"],
            "label": f"Brand: {bbox['label']}",
            "bbox": bbox["bbox"],
            "details": "Potential profiling risk"
        })

    logger.debug(f"Privacy zones: {privacy_zones}")
    return privacy_zones

def analyze_caption(caption):
    """Analyze the tone, intent, and risks in a given caption using spaCy."""
    try:
        doc = nlp(caption)
        tone = "Neutral"
        intent = "Informative"
        risks = []
        for sent in doc.sents:
            if any(token.text.lower() in ['sad', 'sorry', 'hurt'] for token in sent):
                tone = "Emotional"
            if any(token.text.lower() in ['party', 'celebrate'] for token in sent):
                intent = "Celebratory"
            if any(token.text.lower() in ['sarcasm', 'joke'] for token in sent):
                intent = "Sarcastic"
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'PERSON']:
                risks.append((f"Contains {ent.label_.lower()}: {ent.text}", 0.8))
        logger.debug(f"Caption analysis: Tone={tone}, Intent={intent}, Risks={risks}")
        return tone, intent, risks
    except Exception as e:
        logger.error(f"Error in caption analysis: {str(e)}", exc_info=True)
        return "Unknown", "Unknown", []

def assess_risk(exif_data, face_count, demographics, face_qualities, scene_description, scene_objects, geo_locations, ocr_sensitive_data, reverse_search, caption_risks, privacy_zones, brands, documents, qr_bboxes, barcode_bboxes):
    """Assess overall privacy risk based on various analyses."""
    risk_level = "Low"
    risk_reasons = []
    recommendations = []
    markers = []

    if exif_data["status"] == "success":
        metadata = exif_data["data"]
        if "google_maps_url" in metadata:
            risk_level = "High"
            risk_reasons.append(f"EXIF data includes GPS location: {metadata['google_maps_url']}")
            recommendations.append("Remove EXIF metadata to prevent location tracking.")
            markers.append({"type": "exif", "label": "GPS Location", "details": metadata["google_maps_url"]})
        if any(key in metadata["exif"] for key in ["DateTime", "Model", "Make"]):
            risk_level = max(risk_level, "Moderate")
            risk_reasons.append("EXIF metadata could provide contextual information.")
            recommendations.append("Strip EXIF data to reduce traceability.")

    if face_count > 0:
        risk_level = "High"
        risk_reasons.append(f"{face_count} face(s) detected, potential identity risk.")
        recommendations.append("Blur or crop faces to protect identities.")
        has_minor = False
        has_high_quality = False
        for i, demo in enumerate(demographics):
            try:
                age = int(demo['age'])
                if age < 18:
                    has_minor = True
            except (ValueError, TypeError):
                pass
            if face_qualities[i] == 'High':
                has_high_quality = True
            if demo['emotion'] in ['sad', 'fear']:
                risk_level = max(risk_level, "Moderate")
                risk_reasons.append(f"Emotional vulnerability detected: {demo['emotion']}.")
                recommendations.append("Review emotional context before sharing.")
            markers.append({"type": "face", "label": f"Face {i+1}: {demo['age']}y, {demo['gender']}", "details": f"Emotion: {demo['emotion']}, Quality: {face_qualities[i]}"})
        if has_minor:
            risk_level = "High"
            risk_reasons.append("Potential minor detected - family/kids privacy risk.")
            recommendations.append("Avoid sharing images with minors.")
        if has_high_quality:
            risk_level = "High"
            risk_reasons.append("High-quality faces detected - risk for deepfake manipulation or impersonation.")
            recommendations.append("Reduce image quality or blur faces to mitigate deepfake risks.")
        if face_count > 1 and has_minor:
            risk_reasons.append("Multiple faces including minors - potential family exposure.")

    if "bedroom" in scene_description.lower() or "kids" in scene_description.lower() or "office" in scene_description.lower():
        risk_level = max(risk_level, "Moderate")
        risk_reasons.append("Private setting detected (e.g., bedroom/office).")
        recommendations.append("Avoid sharing images from private spaces.")
        markers.append({"type": "scene", "label": "Private Setting", "details": scene_description})

    luxury_objects = [obj for obj in scene_objects if obj[0] in ['car', 'handbag', 'tie', 'wine glass', 'suitcase']]
    workplace_objects = [obj for obj in scene_objects if obj[0] in ['laptop', 'keyboard', 'mouse', 'tv']]
    for obj in scene_objects:
        if obj[0] in ['laptop', 'cell phone', 'book', 'key']:
            risk_level = "High"
            risk_reasons.append(f"Sensitive object: {obj[0]} (Confidence: {obj[1]:.2f}).")
            recommendations.append(f"Blur or remove {obj[0]} before sharing.")
            markers.append({"type": "object", "label": f"Sensitive: {obj[0]}", "details": f"Confidence: {obj[1]:.2f}"})
    if luxury_objects:
        risk_level = max(risk_level, "Moderate")
        risk_reasons.append(f"Luxury items detected: {', '.join([o[0] for o in luxury_objects])} - risk for profiling or targeted ads.")
        recommendations.append("Remove or blur luxury items to avoid profiling.")
    if workplace_objects:
        risk_level = max(risk_level, "Moderate")
        risk_reasons.append(f"Workplace items detected: {', '.join([o[0] for o in workplace_objects])} - corporate data exposure risk.")
        recommendations.append("Avoid sharing workplace-related images.")

    if geo_locations:
        risk_level = max(risk_level, "Moderate")
        risk_reasons.append(f"Potential locations inferred: {', '.join(geo_locations)}.")
        recommendations.append("Avoid sharing images with identifiable landmarks or signs.")
        markers.append({"type": "geo", "label": "Location Inference", "details": ', '.join(geo_locations)})

    if ocr_sensitive_data:
        risk_level = "High"
        for data, label in ocr_sensitive_data:
            risk_reasons.append(f"Sensitive text detected: {data} ({label}).")
            recommendations.append("Blur or remove sensitive text before sharing.")
            markers.append({"type": "text", "label": f"Sensitive Text: {data}", "details": label})
        if any(label in ['AADHAAR', 'PAN', 'PHONE', 'ADDRESS_PIN', 'SIGNATURE', 'LICENSE_PLATE'] for _, label in ocr_sensitive_data):
            risk_reasons.append("Personal data (e.g., Aadhaar/PAN/phone/address/signature/plate) detected - high identity theft risk.")
            recommendations.append("Redact all personal identifiers.")
            risk_reasons.append("Potential compliance violation (e.g., DPDP/GDPR) due to PII leak.")

    if reverse_search["found"]:
        risk_level = max(risk_level, "Moderate")
        risk_reasons.append("Image found on public web.")
        recommendations.append("Avoid reusing publicly available images.")
        markers.append({"type": "reverse", "label": "Public Image", "details": reverse_search["details"]})

    for risk, conf in caption_risks:
        risk_level = max(risk_level, "Moderate")
        risk_reasons.append(f"Caption risk: {risk} (Confidence: {conf:.2f}).")
        recommendations.append("Rephrase caption to avoid sensitive information.")
        markers.append({"type": "caption", "label": "Caption Risk", "details": risk})

    if brands:
        risk_level = max(risk_level, "Moderate")
        risk_reasons.append(f"Brands detected: {', '.join([f'{b[0]} ({b[1]:.2f}, {b[2]})' for b in brands])} - potential profiling risk.")
        recommendations.append("Blur brand logos to avoid targeted advertising.")
        markers.append({"type": "brand", "label": "Brand Detection", "details": ', '.join([f"{b[0]} ({b[1]:.2f}, {b[2]})" for b in brands])})

    if documents:
        risk_level = "High"
        risk_reasons.append(f"Personal documents detected: {', '.join([d[0] for d in documents])} - high risk if leaked.")
        recommendations.append("Do not share images containing documents.")
        markers.append({"type": "document", "label": "Document Detection", "details": ', '.join([f"{d[0]} ({d[1]:.2f})" for d in documents])})
        risk_reasons.append("Confidential corporate/personal data exposure (e.g., invoices/contracts).")
        risk_reasons.append("Compliance risk (GDPR/HIPAA/DPDP) due to sensitive document leak.")

    if qr_bboxes or barcode_bboxes:
        risk_level = "High"
        risk_reasons.append(f"QR/Barcodes detected - potential credentials or data leak.")
        recommendations.append("Blur QR/barcodes to prevent scanning.")
        markers.append({"type": "qr_barcode", "label": "QR/Barcode", "details": f"QR: {len(qr_bboxes)}, Barcodes: {len(barcode_bboxes)}"})

    for zone in privacy_zones:
        markers.append(zone)

    if risk_level == "Low":
        recommendations.append("No significant risks detected, but review before sharing.")

    return {
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "recommendations": recommendations,
        "markers": markers
    }