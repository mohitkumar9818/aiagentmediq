"""
MediQ API Server (v6.0)
Full-featured backend with authentication, database persistence,
report comparison, and enhanced AI analysis endpoints.
"""

import os
import uuid 
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

# Load .env FIRST — before any module that reads os.getenv()
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Ensure Windows terminals can print Unicode log characters from imported modules.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Internal Modules
from ai_engine import analyze_medical_text
from extractors import extract_text
from database import (
    create_user, get_user_by_email, get_user_by_id,
    update_user_login, save_report, get_user_reports,
    get_report_by_uid, get_two_reports, delete_report,
    get_user_report_count
)
from utils import (
    hash_password, verify_password, create_token,
    auth_required, validate_email, validate_password,
    compare_parameters, decode_token
)


# -----------------------------------
# APP INITIALIZATION
# -----------------------------------

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# -----------------------------------
# CONFIGURATION
# -----------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".csv"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Session-scoped telemetry (resets on restart — for monitoring)
SYSTEM_METRICS = {
    "total_files_processed": 0,
    "successful_scans": 0,
    "failed_scans": 0,
    "total_users": 0
}


# -----------------------------------
# HELPERS
# -----------------------------------

def validate_file(file):
    if not file or not file.filename:
        return False, "No file provided."

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {ext}"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > MAX_FILE_SIZE_BYTES:
        return False, "File size exceeds 10MB limit."

    return True, None


def safe_delete(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


# ===================================================
# PUBLIC ROUTES
# ===================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "MediQ AI Analysis API",
        "status": "online",
        "version": "v6.0-enhanced",
        "features": [
            "multi-format-ingestion",
            "dual-ai-engine",
            "user-authentication",
            "report-history",
            "report-comparison",
            "10-day-health-plan",
            "doctor-perspective",
            "organ-system-scores"
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "uptime": "active",
        "version": "v6.0"
    })


# ===================================================
# AUTHENTICATION ROUTES
# ===================================================

@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body required"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    full_name = data.get("full_name", "").strip()

    # Validate
    if not validate_email(email):
        return jsonify({"error": "Invalid email address"}), 400

    is_valid, msg = validate_password(password)
    if not is_valid:
        return jsonify({"error": msg}), 400

    if not full_name or len(full_name) < 2:
        return jsonify({"error": "Full name is required (min 2 characters)"}), 400

    # Check if user exists
    existing = get_user_by_email(email)
    if existing:
        return jsonify({"error": "An account with this email already exists"}), 409

    # Create user
    pw_hash = hash_password(password)
    user_id = create_user(email, pw_hash, full_name)

    if not user_id:
        return jsonify({"error": "Failed to create account"}), 500

    SYSTEM_METRICS["total_users"] += 1

    token = create_token(user_id, email)

    return jsonify({
        "message": "Account created successfully",
        "token": token,
        "user": {
            "id": user_id,
            "email": email,
            "full_name": full_name
        }
    }), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body required"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = get_user_by_email(email)
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    if not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid email or password"}), 401

    update_user_login(user["id"])
    token = create_token(user["id"], email)

    return jsonify({
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"]
        }
    })


@app.route("/auth/me", methods=["GET"])
@auth_required
def get_me():
    user = get_user_by_id(g.user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    report_count = get_user_report_count(g.user_id)

    return jsonify({
        "user": user,
        "stats": {
            "total_reports": report_count
        }
    })


# ===================================================
# UPLOAD & ANALYSIS (AUTHENTICATED)
# ===================================================

@app.route("/upload", methods=["POST"])
@auth_required
def upload():
    start_time = time.time()
    session_id = f"SES-{uuid.uuid4().hex[:8].upper()}"

    if "file" not in request.files:
        SYSTEM_METRICS["failed_scans"] += 1
        return jsonify({"error": "No file received", "session_id": session_id}), 400

    file = request.files["file"]

    is_valid, error = validate_file(file)
    if not is_valid:
        SYSTEM_METRICS["failed_scans"] += 1
        return jsonify({"error": error, "session_id": session_id}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = UPLOAD_DIR / filename

    file.save(filepath)

    try:
        extracted_text = extract_text(filepath)

        if not extracted_text or len(extracted_text.strip()) < 10:
            raise ValueError("Empty or unreadable document")

        ai_result = analyze_medical_text(extracted_text)

        processing_time = round(time.time() - start_time, 2)

        report_uid = f"REP-{uuid.uuid4().hex[:10].upper()}"

        # Save to database
        save_report(
            user_id=g.user_id,
            report_uid=report_uid,
            filename=file.filename,
            file_type=Path(file.filename).suffix[1:].upper(),
            health_score=ai_result["risk_metrics"]["health_score"],
            overall_risk=ai_result["risk_metrics"]["overall_risk"],
            analysis_data=ai_result
        )

        # Attach metadata
        ai_result["meta"] = {
            "report_uid": report_uid,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": file.filename,
            "file_type": Path(file.filename).suffix[1:].upper(),
            "health_score": ai_result["risk_metrics"]["health_score"],
            "overall_risk": ai_result["risk_metrics"]["overall_risk"],
            "processing_time": processing_time
        }

        ai_result["system_stats"] = SYSTEM_METRICS

        SYSTEM_METRICS["total_files_processed"] += 1
        SYSTEM_METRICS["successful_scans"] += 1

        return jsonify(ai_result)

    except Exception as e:
        SYSTEM_METRICS["failed_scans"] += 1
        print(f"[ERROR] {str(e)}")

        # Provide the actual exception message as the main error if it's our own RuntimeError/ValueError
        error_msg = str(e) if isinstance(e, (RuntimeError, ValueError)) else "Internal processing failure"

        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "session_id": session_id
        }), 500

    finally:
        safe_delete(filepath)


# ===================================================
# REPORT HISTORY (AUTHENTICATED)
# ===================================================

@app.route("/reports", methods=["GET"])
@auth_required
def get_reports():
    reports = get_user_reports(g.user_id)
    return jsonify({
        "count": len(reports),
        "reports": reports,
        "metrics": SYSTEM_METRICS
    })


@app.route("/reports/<report_uid>", methods=["GET"])
@auth_required
def get_single_report(report_uid):
    report = get_report_by_uid(report_uid, g.user_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    return jsonify(report)


@app.route("/reports/<report_uid>", methods=["DELETE"])
@auth_required
def remove_report(report_uid):
    deleted = delete_report(report_uid, g.user_id)
    if not deleted:
        return jsonify({"error": "Report not found"}), 404
    return jsonify({"message": "Report deleted successfully"})


# ===================================================
# REPORT COMPARISON (AUTHENTICATED)
# ===================================================

@app.route("/reports/compare", methods=["POST"])
@auth_required
def compare_reports():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    uid_old = data.get("report_old")
    uid_new = data.get("report_new")

    if not uid_old or not uid_new:
        return jsonify({"error": "Both report_old and report_new UIDs are required"}), 400

    report_old, report_new = get_two_reports(uid_old, uid_new, g.user_id)

    if not report_old or not report_new:
        return jsonify({"error": "One or both reports not found"}), 404

    old_data = report_old.get("analysis_data", {}) or {}
    new_data = report_new.get("analysis_data", {}) or {}

    old_params = old_data.get("parameters", [])
    new_params = new_data.get("parameters", [])

    comparison = compare_parameters(old_params, new_params)

    old_score = report_old.get("health_score", 0)
    new_score = report_new.get("health_score", 0)

    # Organ score comparison
    old_organs = old_data.get("organ_scores", {})
    new_organs = new_data.get("organ_scores", {})
    organ_comparison = {}
    for key in ["metabolic", "cardiac", "renal", "hepatic", "hematologic"]:
        organ_comparison[key] = {
            "old": old_organs.get(key, 0),
            "new": new_organs.get(key, 0),
            "change": new_organs.get(key, 0) - old_organs.get(key, 0)
        }

    return jsonify({
        "comparison": {
            "parameters": comparison,
            "health_score": {
                "old": old_score,
                "new": new_score,
                "change": new_score - old_score,
                "trend": "improved" if new_score > old_score
                         else "declined" if new_score < old_score
                         else "stable"
            },
            "organ_scores": organ_comparison,
            "report_old": {
                "uid": uid_old,
                "filename": report_old.get("filename"),
                "date": report_old.get("created_at"),
                "risk": report_old.get("overall_risk")
            },
            "report_new": {
                "uid": uid_new,
                "filename": report_new.get("filename"),
                "date": report_new.get("created_at"),
                "risk": report_new.get("overall_risk")
            }
        },
        "medical_disclaimer": (
            "This comparison is AI-generated and for informational purposes only. "
            "Values may contain inaccuracies. Always consult a healthcare professional."
        )
    })


# ===================================================
# LEGACY HISTORY ENDPOINT (for backward compatibility)
# ===================================================

@app.route("/history", methods=["GET"])
def legacy_history():
    """Legacy endpoint — returns empty for unauthenticated requests."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        payload = decode_token(token)
        if payload:
            reports = get_user_reports(payload["user_id"])
            return jsonify({
                "count": len(reports),
                "reports": reports,
                "metrics": SYSTEM_METRICS
            })

    return jsonify({
        "count": 0,
        "reports": [],
        "metrics": SYSTEM_METRICS
    })


# -----------------------------------
# ENTRY POINT (Local only)
# -----------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 MediQ API v6.0 starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
