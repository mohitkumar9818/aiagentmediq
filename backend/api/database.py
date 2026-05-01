"""
MediQ Database Layer
SQLite-backed persistent storage for users, reports, and health plans.
Designed for local/demo use — easily upgradeable to PostgreSQL/Supabase.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "mediq.db"


# ======================================================
# CONNECTION
# ======================================================

def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ======================================================
# SCHEMA INITIALIZATION
# ======================================================

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            report_uid TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            health_score INTEGER DEFAULT 0,
            overall_risk TEXT DEFAULT 'unknown',
            analysis_data TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_reports_user ON reports(user_id);
        CREATE INDEX IF NOT EXISTS idx_reports_uid ON reports(report_uid);
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized at:", DB_PATH)


# ======================================================
# USER OPERATIONS
# ======================================================

def create_user(email: str, password_hash: str, full_name: str) -> int | None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)",
            (email, password_hash, full_name)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, full_name, created_at, last_login FROM users WHERE id = ?",
        (user_id,)
    )
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def update_user_login(user_id: int):
    conn = get_connection()
    conn.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), user_id)
    )
    conn.commit()
    conn.close()


# ======================================================
# REPORT OPERATIONS
# ======================================================

def save_report(user_id: int, report_uid: str, filename: str, file_type: str,
                health_score: int, overall_risk: str, analysis_data: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO reports 
           (user_id, report_uid, filename, file_type, health_score, overall_risk, analysis_data)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, report_uid, filename, file_type,
         health_score, overall_risk, json.dumps(analysis_data))
    )
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()
    return report_id


def get_user_reports(user_id: int, limit: int = 50) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, report_uid, filename, file_type, health_score, overall_risk, created_at
           FROM reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
        (user_id, limit)
    )
    reports = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return reports


def get_report_by_uid(report_uid: str, user_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM reports WHERE report_uid = ? AND user_id = ?",
        (report_uid, user_id)
    )
    report = cursor.fetchone()
    conn.close()
    if report:
        r = dict(report)
        r['analysis_data'] = json.loads(r['analysis_data']) if r['analysis_data'] else None
        return r
    return None


def get_two_reports(uid_1: str, uid_2: str, user_id: int) -> tuple:
    """Fetch two reports for comparison."""
    r1 = get_report_by_uid(uid_1, user_id)
    r2 = get_report_by_uid(uid_2, user_id)
    return r1, r2


def delete_report(report_uid: str, user_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM reports WHERE report_uid = ? AND user_id = ?",
        (report_uid, user_id)
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_user_report_count(user_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM reports WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()["cnt"]
    conn.close()
    return count


# Initialize database on import
init_db()
