# exif_utils.py

import os
import logging
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime

logger = logging.getLogger(__name__)

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
            gps_coords["lat"][0][0] / gps_coords["lat"][0][1],
            gps_coords["lat"][1][0] / gps_coords["lat"][1][1],
            gps_coords["lat"][2][0] / gps_coords["lat"][2][1],
            gps_coords["lat_ref"]
        )
        dec_deg_lon = convert_decimal_degrees(
            gps_coords["lon"][0][0] / gps_coords["lon"][0][1],
            gps_coords["lon"][1][0] / gps_coords["lon"][1][1],
            gps_coords["lon"][2][0] / gps_coords["lon"][2][1],
            gps_coords["lon_ref"]
        )
        if dec_deg_lat is None or dec_deg_lon is None:
            return None
        return f"https://maps.google.com/?q={dec_deg_lat:.6f},{dec_deg_lon:.6f}"
    except (KeyError, ValueError) as e:
        logger.debug(f"Error creating Google Maps URL: {e}")
        return None

def extract_exif_data(image_path):
    """Extract EXIF metadata from the image, including GPS data."""
    try:
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
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
                metadata["gps_coords"] = gps_coords  # Added to support geo_inference
            
            return {"status": "success", "data": metadata}
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e), "data": {"filename": os.path.basename(image_path)}}