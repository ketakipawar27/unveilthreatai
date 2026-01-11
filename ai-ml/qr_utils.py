import os
import logging
import cv2

logger = logging.getLogger(__name__)

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