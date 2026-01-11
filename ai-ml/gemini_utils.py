import os
import json
import time
import logging
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ===================== ENV =====================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Check .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

# ===================== RATE LIMIT CONTROL =====================
LAST_GEMINI_CALL = 0
GEMINI_COOLDOWN = 45  # seconds (free tier safe)

# ===================== MAIN =====================
def gemini_place_detection(image_path):
    """
    Gemini Vision landmark detection (rate-limit safe).
    Returns structured geo evidence or None.
    """
    global LAST_GEMINI_CALL

    # ⛔ Cooldown protection
    if time.time() - LAST_GEMINI_CALL < GEMINI_COOLDOWN:
        logger.info("Gemini skipped (cooldown active)")
        return None

    try:
        LAST_GEMINI_CALL = time.time()

        img = Image.open(image_path).convert("RGB")

        prompt = """
You are a geographic landmark expert.

Identify the real-world landmark ONLY if:
- The landmark is globally recognizable
- Confidence is above 90%
- The landmark cannot be confused with similar places

Respond ONLY in JSON:
{
  "location": "<landmark, city, country>",
  "confidence": <float 0-1>
}

If unsure or ambiguous:
{ "location": null, "confidence": 0 }
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        if not response.text:
            return None

        data = json.loads(response.text)
        location = data.get("location")
        confidence = float(data.get("confidence", 0))

        if location and confidence >= 0.90:
            return {
                "location": location,
                "source": "gemini_vision",
                "confidence": round(confidence, 2),
                "status": "confirmed"
            }

    except Exception as e:
        logger.warning(f"Gemini place detection failed: {e}")

    return None
