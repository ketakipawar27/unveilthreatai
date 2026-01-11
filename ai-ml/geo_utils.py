import os
import logging
from exif_utils import extract_exif_data, convert_decimal_degrees
from gemini_utils import gemini_place_detection
from geopy.geocoders import Nominatim

logger = logging.getLogger(__name__)

GPS_CONFIDENCE = 0.97


def geo_inference(image_path):
    """
    STRICT GEO POLICY:
    1️⃣ EXIF GPS → ONLY SOURCE if present
    2️⃣ ELSE Gemini CNN visual inference
    3️⃣ ELSE nothing
    """

    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            return []

        # =====================================================
        # 1️⃣ EXIF GPS — ABSOLUTE AUTHORITY
        # =====================================================
        exif_data = extract_exif_data(image_path)

        if exif_data["status"] == "success" and "gps_coords" in exif_data["data"]:
            try:
                gps = exif_data["data"]["gps_coords"]

                lat = convert_decimal_degrees(
                    gps["lat"][0][0] / gps["lat"][0][1],
                    gps["lat"][1][0] / gps["lat"][1][1],
                    gps["lat"][2][0] / gps["lat"][2][1],
                    gps["lat_ref"]
                )
                lon = convert_decimal_degrees(
                    gps["lon"][0][0] / gps["lon"][0][1],
                    gps["lon"][1][0] / gps["lon"][1][1],
                    gps["lon"][2][0] / gps["lon"][2][1],
                    gps["lon_ref"]
                )

                if lat is not None and lon is not None:
                    geolocator = Nominatim(user_agent="digital_domino")
                    location = geolocator.reverse((lat, lon), language="en")

                    if location and location.address:
                        return [{
                            "location": location.address,
                            "source": "exif_gps",
                            "confidence": GPS_CONFIDENCE,
                            "status": "confirmed"
                        }]

            except Exception as e:
                logger.warning(f"EXIF GPS failed: {e}")

        # =====================================================
        # 2️⃣ CNN VISUAL INFERENCE (Gemini)
        # =====================================================
        gemini_result = gemini_place_detection(image_path)

        if gemini_result:
            return [{
                "location": gemini_result["location"],
                "source": "visual_analysis",
                "confidence": gemini_result["confidence"],
                "status": "likely"
            }]

        # =====================================================
        # 3️⃣ NOTHING
        # =====================================================
        logger.info(f"No location inferred for {image_path}")
        return []

    except Exception as e:
        logger.error(f"Geo inference error: {e}", exc_info=True)
        return []
