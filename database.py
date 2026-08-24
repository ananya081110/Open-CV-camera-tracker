import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "logs" / "camera_events.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                person_id INTEGER,
                event_type TEXT NOT NULL,
                activity TEXT,
                details TEXT,
                image_path TEXT
            )
        """)

        conn.commit()


def log_event(
    person_id,
    event_type,
    activity=None,
    details="",
    image_path=None
):
    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    with get_connection() as conn:

        cursor = conn.execute("""
            INSERT INTO events (
                timestamp,
                person_id,
                event_type,
                activity,
                details,
                image_path
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            person_id,
            event_type,
            activity,
            details,
            image_path
        ))

        conn.commit()

        return cursor.lastrowid


def get_recent_events(limit=20):

    with get_connection() as conn:

        return conn.execute("""
            SELECT *
            FROM events
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()