"""
MediQ Utilities
JWT authentication, validation, and helper functions.
"""

import os
import jwt
import functools
from pathlib import Path
from datetime import datetime, timezone, timedelta
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Always load .env using absolute path so JWT_SECRET is found regardless
# of which directory the server is launched from.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ======================================================
# JWT CONFIGURATION
# ======================================================

JWT_SECRET: str = os.getenv("JWT_SECRET") or "mediq-dev-secret-change-in-production-2026"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 72  # Token valid for 3 days

print(f"[JWT] Secret loaded: {str(JWT_SECRET)[:8]}... (length={len(JWT_SECRET)})")  # type: ignore[index]


# ======================================================
# PASSWORD HASHING
# ======================================================

def hash_password(password: str) -> str:
    return generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


# ======================================================
# JWT TOKEN MANAGEMENT
# ======================================================

def create_token(user_id: int, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ======================================================
# AUTH MIDDLEWARE (DECORATOR)
# ======================================================

def auth_required(f):
    """Decorator to protect routes that require authentication."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization token required"}), 401

        token = auth_header.split(" ")[1]
        payload = decode_token(token)

        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        # Attach user info to request context
        g.user_id = payload["user_id"]
        g.user_email = payload["email"]

        return f(*args, **kwargs)

    return decorated


# ======================================================
# VALIDATION HELPERS
# ======================================================

def validate_email(email: str) -> bool:
    """Basic email validation."""
    if not email or "@" not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False
    if "." not in parts[1]:
        return False
    return True


def validate_password(password: str) -> tuple[bool, str]:
    """Password strength validation."""
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Remove potentially dangerous characters from filenames."""
    return "".join(c for c in filename if c.isalnum() or c in "._- ")


# ======================================================
# COMPARISON HELPERS
# ======================================================

def compare_parameters(params_old: list, params_new: list) -> list:
    """Compare two sets of biomarker parameters and produce a diff."""
    old_map = {p.get("name", "").lower(): p for p in params_old}
    new_map = {p.get("name", "").lower(): p for p in params_new}

    all_keys = set(list(old_map.keys()) + list(new_map.keys()))
    comparison = []

    for key in sorted(all_keys):
        old = old_map.get(key)
        new = new_map.get(key)

        if old and new:
            try:
                old_val = float(str(old.get("value", "0")).replace(",", ""))
                new_val = float(str(new.get("value", "0")).replace(",", ""))
                change = int((new_val - old_val) * 100) / 100.0
                change_pct = int((change / old_val * 100) * 10) / 10.0 if old_val != 0 else 0.0
            except (ValueError, TypeError):
                change = 0
                change_pct = 0

            comparison.append({
                "name": new.get("name", key),
                "old_value": old.get("value", "N/A"),
                "new_value": new.get("value", "N/A"),
                "unit": new.get("unit", ""),
                "old_status": old.get("status", "normal"),
                "new_status": new.get("status", "normal"),
                "change": change,
                "change_percent": change_pct,
                "trend": "improved" if (new.get("status") == "normal" and old.get("status") != "normal")
                         else "declined" if (new.get("status") != "normal" and old.get("status") == "normal")
                         else "stable"
            })
        elif new:
            comparison.append({
                "name": new.get("name", key),
                "old_value": "N/A",
                "new_value": new.get("value", "N/A"),
                "unit": new.get("unit", ""),
                "old_status": "N/A",
                "new_status": new.get("status", "normal"),
                "change": 0,
                "change_percent": 0,
                "trend": "new"
            })

    return comparison
