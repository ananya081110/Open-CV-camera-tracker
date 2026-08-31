import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).resolve().parent / "logs" / "users.db"


def init_auth_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create a default admin only if no users exist.
        existing = conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        if existing == 0:
            conn.execute(
                """
                INSERT INTO users
                (username, password_hash, role)
                VALUES (?, ?, ?)
                """,
                (
                    "admin",
                    generate_password_hash("admin123"),
                    "admin",
                ),
            )

        conn.commit()


def authenticate(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT username, password_hash, role
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if row is None:
        return None

    if not check_password_hash(
        row[1],
        password
    ):
        return None

    return {
        "username": row[0],
        "role": row[2],
    }


def create_user(username, password, role="viewer"):
    if role not in ("admin", "viewer"):
        raise ValueError("Invalid role")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO users
            (username, password_hash, role)
            VALUES (?, ?, ?)
            """,
            (
                username,
                generate_password_hash(password),
                role,
            ),
        )
        conn.commit()


def change_password(username, new_password):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE username = ?
            """,
            (
                generate_password_hash(new_password),
                username,
            ),
        )
        conn.commit()
