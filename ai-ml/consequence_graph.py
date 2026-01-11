# consequence_graph.py

import uuid


def _id(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def build_consequence_graph(ai_output, input_type="file"):
    """
    Builds a dynamic consequence graph from AI output.
    Enforces strict deduplication of signal nodes.
    """

    nodes = []
    edges = []

    # ---------------- ROOT ----------------
    root_id = "root_file"
    nodes.append({
        "id": root_id,
        "label": "Uploaded Image / Document",
        "type": "root",
        "level": 0
    })

    # Track created signals to avoid duplicates
    signal_nodes = {}

    def add_node(parent_id, key, label, node_type, level):
        """
        Add node ONLY if key not already created.
        """
        if key in signal_nodes:
            return signal_nodes[key]

        node_id = f"{_id(label)}_{str(uuid.uuid4())[:6]}"
        nodes.append({
            "id": node_id,
            "label": label,
            "type": node_type,
            "level": level
        })

        edges.append({
            "from": parent_id,
            "to": node_id
        })

        signal_nodes[key] = node_id
        return node_id

    # =====================================================
    # LEVEL 1 — SIGNALS
    # =====================================================

    # ---------------- IMAGE RULES ----------------
    if input_type == "image":

        # Sensitive documents in image
        for doc in ai_output.get("documents", []):
            if doc.get("tier") == "HIGH":
                add_node(
                    root_id,
                    f"doc_{doc['document']}",
                    f"{doc['document']} Detected",
                    "signal",
                    1
                )

        # OCR sensitive text
        if ai_output.get("ocr_sensitive_data"):
            add_node(
                root_id,
                "ocr",
                "Sensitive Text Detected",
                "signal",
                1
            )

        # Face
        if ai_output.get("face_count", 0) > 0:
            add_node(
                root_id,
                "face",
                "Human Face Detected",
                "signal",
                1
            )

        # Location
        if ai_output.get("geo_locations"):
            add_node(
                root_id,
                "location",
                "Location Information Found",
                "signal",
                1
            )

        # Metadata
        exif = ai_output.get("exif", {})
        if exif and exif.get("status") != "no_exif":
            add_node(
                root_id,
                "metadata",
                "Metadata Present",
                "signal",
                1
            )

    # ---------------- DOCUMENT RULES ----------------
    if input_type == "document":
        details = ai_output.get("details", {})

        # Sensitive structured data
        for item in details.get("sensitive_data", []):
            dtype = item.get("type")
            if dtype:
                add_node(
                    root_id,
                    f"sensitive_{dtype}",
                    f"{dtype} Detected",
                    "signal",
                    1
                )

        # External links
        if details.get("links"):
            add_node(
                root_id,
                "links",
                "External Links Found",
                "signal",
                1
            )

        # Embedded images
        if details.get("embedded_images_count", 0) > 0:
            add_node(
                root_id,
                "embedded_images",
                "Embedded Images Found",
                "signal",
                1
            )

        # Face inside document images
        for img in details.get("image_analysis", []):
            if img.get("face_count", 0) > 0:
                add_node(
                    root_id,
                    "face",
                    "Human Face Detected (Document Image)",
                    "signal",
                    1
                )
                break

        # Sensitive documents inside images
        for img in details.get("image_analysis", []):
            for doc in img.get("documents", []):
                if doc.get("tier") == "HIGH":
                    add_node(
                        root_id,
                        f"doc_{doc['document']}",
                        f"{doc['document']} Detected",
                        "signal",
                        1
                    )

    # =====================================================
    # LEVEL 2 — RISKS
    # =====================================================
    risk_nodes = {}

    def add_risk(parent_key, risk_key, label):
        if parent_key in signal_nodes and risk_key not in risk_nodes:
            node_id = f"{_id(label)}_{str(uuid.uuid4())[:6]}"
            nodes.append({
                "id": node_id,
                "label": label,
                "type": "risk",
                "level": 2
            })
            edges.append({
                "from": signal_nodes[parent_key],
                "to": node_id
            })
            risk_nodes[risk_key] = node_id

    # Identity / financial
    if any(k.startswith("doc_") for k in signal_nodes):
        add_risk("doc_Passport", "identity", "Identity / Financial Exposure")

    # Sensitive data leakage
    if any(k.startswith("sensitive_") for k in signal_nodes):
        add_risk("sensitive_EMAIL", "data_leak", "Sensitive Data Leakage Risk")

    # Privacy
    if "face" in signal_nodes:
        add_risk("face", "privacy", "Personal Privacy Violation")

    # Phishing
    if "links" in signal_nodes:
        add_risk("links", "phishing", "Phishing or Malicious Redirect")

    # Metadata profiling
    if "metadata" in signal_nodes:
        add_risk("metadata", "profiling", "Device / Profile Fingerprinting")

    # =====================================================
    # LEVEL 3 — CONSEQUENCES
    # =====================================================
    def add_consequence(parent_id, label):
        node_id = f"{_id(label)}_{str(uuid.uuid4())[:6]}"
        nodes.append({
            "id": node_id,
            "label": label,
            "type": "consequence",
            "level": 3
        })
        edges.append({
            "from": parent_id,
            "to": node_id
        })

    if "identity" in risk_nodes:
        add_consequence(risk_nodes["identity"], "Identity Theft")
        add_consequence(risk_nodes["identity"], "Financial Fraud")

    if "data_leak" in risk_nodes:
        add_consequence(risk_nodes["data_leak"], "Account Compromise")

    if "privacy" in risk_nodes:
        add_consequence(risk_nodes["privacy"], "Reputation Damage")

    if "profiling" in risk_nodes:
        add_consequence(risk_nodes["profiling"], "Targeted Surveillance")

    if "phishing" in risk_nodes:
        add_consequence(risk_nodes["phishing"], "Credential Theft")

    # =====================================================
    return {
        "nodes": nodes,
        "edges": edges
    }
