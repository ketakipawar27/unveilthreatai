# document_loader.py

import os
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------
# SUPPORTED DOCUMENT TYPES
# --------------------------------------------------
PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx"}
PPT_EXTENSIONS = {".pptx"}
TEXT_EXTENSIONS = {".txt"}

ALL_SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS |
    WORD_EXTENSIONS |
    PPT_EXTENSIONS |
    TEXT_EXTENSIONS
)

# --------------------------------------------------
# DOCUMENT TYPE DETECTOR
# --------------------------------------------------
def detect_document_type(file_path):
    """
    Detect document type based on extension.

    Returns:
    {
        "valid": bool,
        "doc_type": str | None,
        "filename": str | None,
        "reason": str
    }
    """

    try:
        if not file_path:
            return {
                "valid": False,
                "doc_type": None,
                "filename": None,
                "reason": "Empty file path"
            }

        file_path = os.path.normpath(file_path)
        filename = os.path.basename(file_path)   # ✅ DEFINED FIRST

        if not os.path.exists(file_path):
            return {
                "valid": False,
                "doc_type": None,
                "filename": filename,
                "reason": "File does not exist"
            }

        if os.path.getsize(file_path) == 0:
            return {
                "valid": False,
                "doc_type": None,
                "filename": filename,
                "reason": "Empty file"
            }

        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in ALL_SUPPORTED_EXTENSIONS:
            return {
                "valid": False,
                "doc_type": None,
                "filename": filename,
                "reason": f"Unsupported file type: {ext}"
            }

        if ext in PDF_EXTENSIONS:
            doc_type = "pdf"
        elif ext in WORD_EXTENSIONS:
            doc_type = "docx"
        elif ext in PPT_EXTENSIONS:
            doc_type = "pptx"
        elif ext in TEXT_EXTENSIONS:
            doc_type = "txt"
        else:
            doc_type = None

        # ✅ ASCII-safe log (no Unicode arrows)
        logger.info(f"Document detected: {doc_type} -> {filename}")

        return {
            "valid": True,
            "doc_type": doc_type,
            "filename": filename,
            "reason": "Supported document"
        }

    except Exception as e:
        logger.error(
            f"Document type detection failed for {file_path}: {e}",
            exc_info=True
        )
        return {
            "valid": False,
            "doc_type": None,
            "filename": None,
            "reason": "Internal error during detection"
        }
