import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from camera_control import load_camera_control, save_camera_control

from auth import (
    authenticate,
    init_auth_db,
    create_user,
    change_password,
)


# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
EVENT_DB_PATH = ROOT / "logs" / "camera_events.db"
AUTH_DB_PATH = ROOT / "logs" / "users.db"

st.set_page_config(
    page_title="AI Camera Tracker | Command Center",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

for key, default in {
    "authenticated": False,
    "username": None,
    "role": None,
    "theme": "Dark",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# DYNAMIC THEME SYSTEM (CSS)
# ============================================================

is_dark = st.session_state.theme == "Dark"

if is_dark:
    theme_vars = """
    --bg-base:        #080c16;
    --card-bg:        rgba(14, 21, 38, 0.72);
    --card-border:    rgba(0, 242, 254, 0.14);
    --neon-cyan:      #00f2fe;
    --neon-blue:      #38bdf8;
    --neon-green:     #10b981;
    --neon-red:       #ef4444;
    --neon-amber:     #f59e0b;
    --text-primary:   #f8fafc;
    --text-muted:     #94a3b8;
    --sidebar-bg:     linear-gradient(180deg, #070a14 0%, #0a0f1d 100%);
    --sidebar-border: rgba(255, 255, 255, 0.08);
    --shadow-card:    0 8px 32px 0 rgba(0, 0, 0, 0.45);
    --radial-1:       rgba(0, 242, 254, 0.08);
    --radial-2:       rgba(56, 189, 248, 0.06);
    --grid-line:      rgba(255, 255, 255, 0.015);
    --tab-hover:      rgba(0, 242, 254, 0.06);
    --tab-active:     rgba(0, 242, 254, 0.09);
    --input-card-bg:  rgba(14, 21, 38, 0.5);
    --status-bg:      rgba(16, 185, 129, 0.12);
    --status-border:  rgba(16, 185, 129, 0.35);
    """
    chart_c1 = "#00f2fe"
    chart_c2 = "#38bdf8"
    chart_c3 = "#10b981"
else:
    theme_vars = """
    --bg-base:        #f4f6fb;
    --card-bg:        rgba(255, 255, 255, 0.92);
    --card-border:    rgba(15, 23, 42, 0.08);
    --neon-cyan:      #0284c7;
    --neon-blue:      #0369a1;
    --neon-green:     #059669;
    --neon-red:       #dc2626;
    --neon-amber:     #d97706;
    --text-primary:   #0f172a;
    --text-muted:     #64748b;
    --sidebar-bg:     linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
    --sidebar-border: rgba(0, 0, 0, 0.08);
    --shadow-card:    0 4px 20px 0 rgba(15, 23, 42, 0.06);
    --radial-1:       rgba(2, 132, 199, 0.05);
    --radial-2:       rgba(14, 165, 233, 0.04);
    --grid-line:      rgba(15, 23, 42, 0.03);
    --tab-hover:      rgba(2, 132, 199, 0.06);
    --tab-active:     rgba(2, 132, 199, 0.10);
    --input-card-bg:  rgba(255, 255, 255, 0.85);
    --status-bg:      rgba(5, 150, 105, 0.12);
    --status-border:  rgba(5, 150, 105, 0.35);
    """
    chart_c1 = "#0284c7"
    chart_c2 = "#0369a1"
    chart_c3 = "#059669"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
    {theme_vars}
    --radius-lg:     16px;
    --radius-md:     12px;
    --radius-sm:     8px;
}}

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    color: var(--text-primary) !important;
}}

.stApp {{
    background-color: var(--bg-base);
    background-image:
        radial-gradient(1200px 500px at 15% -5%, var(--radial-1), transparent 60%),
        radial-gradient(1000px 450px at 85% 105%, var(--radial-2), transparent 60%),
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
}}

.main .block-container {{
    max-width: 1520px;
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
}}

/* Monospace treatment for data, IDs & metrics */
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] *,
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}}

/* ---------- Sidebar Styling ---------- */
section[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border);
}}

section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{
    color: var(--text-muted) !important;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: var(--text-primary) !important;
    font-weight: 700;
}}

section[data-testid="stSidebar"] .stButton button {{
    width: 100%;
    border-radius: var(--radius-sm);
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    color: var(--text-primary) !important;
    font-weight: 600;
    transition: all 0.2s ease;
}}

section[data-testid="stSidebar"] .stButton button:hover {{
    border-color: var(--neon-cyan);
    color: var(--neon-cyan) !important;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 8px;
}}

.sidebar-brand-mark {{
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #00f2fe, #0284c7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
    box-shadow: 0 0 14px rgba(0, 242, 254, 0.4);
    flex-shrink: 0;
}}

.sidebar-brand-name {{
    font-size: 16px;
    font-weight: 800;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
}}

.sidebar-brand-sub {{
    font-size: 10px;
    color: var(--neon-cyan) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-top: 2px;
}}

.sidebar-meta-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    font-size: 11.5px;
    line-height: 1.7;
    box-shadow: var(--shadow-card);
}}

/* ---------- Header Panel ---------- */
.app-header {{
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
    padding: 20px 24px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 18px;
    position: relative;
    overflow: hidden;
}}

.app-header::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--neon-cyan), transparent);
}}

.app-header-mark {{
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: linear-gradient(135deg, #00f2fe, #0284c7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.45);
}}

.app-title {{
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary) !important;
}}

.app-subtitle {{
    margin-top: 3px;
    color: var(--text-muted) !important;
    font-size: 13px;
}}

.section-title {{
    font-size: 19px;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin-top: 6px;
    margin-bottom: 2px;
    padding-left: 12px;
    border-left: 3px solid var(--neon-cyan);
    color: var(--text-primary) !important;
}}

.section-subtitle {{
    color: var(--text-muted) !important;
    font-size: 13px;
    margin-bottom: 18px;
    padding-left: 12px;
}}

/* ---------- Status Beacons ---------- */
.status-online {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--status-bg);
    color: var(--neon-green) !important;
    border: 1px solid var(--status-border);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

.status-online::before {{
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--neon-green);
    box-shadow: 0 0 10px var(--neon-green);
    animation: pulse 1.8s infinite ease-in-out;
}}

@keyframes pulse {{
    0% {{ transform: scale(0.9); opacity: 0.7; }}
    50% {{ transform: scale(1.3); opacity: 1; filter: drop-shadow(0 0 4px var(--neon-green)); }}
    100% {{ transform: scale(0.9); opacity: 0.7; }}
}}

/* ---------- Alert Banners ---------- */
.alert-banner {{
    border-radius: var(--radius-md);
    padding: 13px 18px;
    margin: 12px 0 18px 0;
    font-size: 13.5px;
    font-weight: 550;
    line-height: 1.5;
    border-left-width: 4px;
    border-left-style: solid;
    backdrop-filter: blur(10px);
}}

.alert-danger {{
    background: rgba(239, 68, 68, 0.12);
    border-color: var(--neon-red);
    color: var(--neon-red) !important;
}}

.alert-warning {{
    background: rgba(245, 158, 11, 0.12);
    border-color: var(--neon-amber);
    color: var(--neon-amber) !important;
}}

.alert-success {{
    background: rgba(16, 185, 129, 0.12);
    border-color: var(--neon-green);
    color: var(--neon-green) !important;
}}

.alert-info {{
    background: rgba(56, 189, 248, 0.12);
    border-color: var(--neon-blue);
    color: var(--neon-blue) !important;
}}

/* ---------- KPI Grid ---------- */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 8px;
}}

.kpi-card {{
    background: var(--card-bg);
    backdrop-filter: blur(14px);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 15px 16px;
    position: relative;
    overflow: hidden;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-card);
}}

.kpi-card:hover {{
    border-color: var(--neon-cyan);
    transform: translateY(-2px);
}}

.kpi-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-bar, var(--neon-cyan));
}}

.kpi-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin-bottom: 6px;
}}

.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary) !important;
    line-height: 1;
}}

/* ---------- Security Incident Cards ---------- */
.security-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin-bottom: 12px;
    border-left-width: 5px;
    border-left-style: solid;
    box-shadow: var(--shadow-card);
}}

.security-critical {{ border-left-color: var(--neon-red); }}
.security-high     {{ border-left-color: #ea580c; }}
.security-medium   {{ border-left-color: var(--neon-amber); }}
.security-normal   {{ border-left-color: var(--neon-green); }}

/* ---------- Login Shell ---------- */
.login-shell {{
    max-width: 440px;
    margin: 5rem auto 0 auto;
}}

.login-card {{
    background: var(--card-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
    padding: 30px 32px 24px 32px;
    text-align: left;
    position: relative;
    overflow: hidden;
}}

.login-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--neon-cyan), #0284c7);
}}

.login-mark {{
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: linear-gradient(135deg, #00f2fe, #0284c7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 14px;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.4);
}}

.login-title {{
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--text-primary) !important;
}}

.login-subtitle {{
    color: var(--text-muted) !important;
    margin-bottom: 6px;
    font-size: 13px;
}}

/* ---------- Tabs Styling ---------- */
div[data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid var(--card-border) !important;
    background: transparent;
}}

button[data-baseweb="tab"] {{
    font-weight: 600;
    font-size: 13px;
    color: var(--text-muted) !important;
    padding: 10px 18px;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    background: transparent;
    transition: all 0.2s ease;
}}

button[data-baseweb="tab"]:hover {{
    background: var(--tab-hover);
    color: var(--neon-cyan) !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text-primary) !important;
    background: var(--tab-active) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
}}

div[data-baseweb="tab-highlight"] {{
    background: var(--neon-cyan) !important;
    height: 2px !important;
}}

/* ---------- Buttons & Inputs ---------- */
.stButton > button, .stFormSubmitButton > button {{
    border-radius: var(--radius-sm);
    font-weight: 650;
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(14, 165, 233, 0.3);
    transition: transform 0.1s ease, filter 0.15s ease;
}}

.stButton > button:hover, .stFormSubmitButton > button:hover {{
    filter: brightness(1.15);
}}

.stButton > button:active, .stFormSubmitButton > button:active {{
    transform: scale(0.98);
}}

/* Dataframe containers */
div[data-testid="stDataFrame"] {{
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    overflow: hidden;
    background: var(--input-card-bg);
}}

hr {{
    border-color: var(--card-border) !important;
    margin: 20px 0 !important;
}}

.footer {{
    color: var(--text-muted) !important;
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    text-align: center;
    padding-top: 36px;
    padding-bottom: 6px;
}}

@media (max-width: 900px) {{
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# AUTH DATABASE / AUDIT LOGGING
# ============================================================

init_auth_db()


def auth_connection():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(AUTH_DB_PATH)


def init_audit_db():
    """Create an audit trail without changing the existing users table."""
    try:
        with auth_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    role TEXT,
                    action TEXT NOT NULL,
                    target TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
    except Exception:
        pass


def write_audit(action, username=None, role=None, target=None, details=None):
    """Best-effort audit logging; never break the dashboard."""
    try:
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs
                (username, role, action, target, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    username,
                    role,
                    action,
                    target,
                    details,
                ),
            )
            conn.commit()
    except Exception:
        pass


def load_audit_logs(limit=500):
    try:
        with auth_connection() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    id,
                    timestamp,
                    username,
                    role,
                    action,
                    target,
                    details
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(int(limit),),
            )
    except Exception:
        return pd.DataFrame()


def load_users():
    try:
        with auth_connection() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    id,
                    username,
                    role,
                    created_at
                FROM users
                ORDER BY id ASC
                """,
                conn,
            )
    except Exception:
        return pd.DataFrame()


def delete_user(username):
    """
    Optional administrative user removal.
    Never allow the currently logged-in account to delete itself.
    """
    try:
        with auth_connection() as conn:
            conn.execute(
                "DELETE FROM users WHERE username = ?",
                (username,),
            )
            conn.commit()
        return True, None
    except Exception as exc:
        return False, str(exc)


init_audit_db()


# ============================================================
# LOGIN
# ============================================================

if not st.session_state.authenticated:
    st.markdown(
        '<div class="login-shell">',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="login-card">
    <div class="login-mark">📹</div>
    <div class="login-title">AI Camera Tracker</div>
    <div class="login-subtitle">
        Secure Operations Command Center
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username",
            autocomplete="username",
        )

        password = st.text_input(
            "Password",
            type="password",
            autocomplete="current-password",
        )

        submitted = st.form_submit_button(
            "Authorize Access",
            use_container_width=True,
        )

    if submitted:
        clean_username = username.strip()

        user = authenticate(
            clean_username,
            password,
        )

        if user is None:
            write_audit(
                "LOGIN_FAILED",
                username=clean_username or None,
                details="Invalid username or password",
            )
            st.error("Invalid username or password.")
        else:
            st.session_state.authenticated = True
            st.session_state.username = user["username"]
            st.session_state.role = user["role"]

            write_audit(
                "LOGIN",
                username=user["username"],
                role=user["role"],
                details="Successful dashboard login",
            )

            st.rerun()

    st.info(
        "First-time local setup: admin / admin123. "
        "Change the password after first login."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# CURRENT USER
# ============================================================

CURRENT_USER = st.session_state.username
CURRENT_ROLE = str(st.session_state.role or "viewer").lower()
IS_ADMIN = CURRENT_ROLE == "admin"


# ============================================================
# DATABASE
# ============================================================

def get_event_connection():
    return sqlite3.connect(EVENT_DB_PATH)


@st.cache_data(ttl=3)
def load_events():
    if not EVENT_DB_PATH.exists():
        return pd.DataFrame()

    try:
        with get_event_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    id,
                    timestamp,
                    person_id,
                    event_type,
                    activity,
                    details,
                    image_path
                FROM events
                ORDER BY timestamp ASC, id ASC
                """,
                conn,
            )
    except Exception as exc:
        st.error(f"Could not read the event database: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )
    df["person_id"] = pd.to_numeric(
        df["person_id"],
        errors="coerce",
    )
    df["event_type"] = (
        df["event_type"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )
    df["activity"] = (
        df["activity"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )
    df["details"] = (
        df["details"]
        .fillna("")
        .astype(str)
    )
    df["image_path"] = (
        df["image_path"]
        .fillna("")
        .astype(str)
    )

    return df


def table_exists(table_name):
    if not EVENT_DB_PATH.exists():
        return False

    try:
        with get_event_connection() as conn:
            row = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table_name,),
            ).fetchone()
            return row is not None
    except Exception:
        return False


def event_db_tables():
    if not EVENT_DB_PATH.exists():
        return []

    try:
        with get_event_connection() as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                ORDER BY name
                """
            ).fetchall()
        return [row[0] for row in rows]
    except Exception:
        return []


@st.cache_data(ttl=3)
def load_security_alerts():
    if not table_exists("security_alerts"):
        return pd.DataFrame()

    try:
        with get_event_connection() as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM security_alerts
                ORDER BY id DESC
                """,
                conn,
            )
    except Exception as exc:
        st.error(f"Could not read security alerts: {exc}")
        return pd.DataFrame()


def security_table_columns():
    if not table_exists("security_alerts"):
        return []

    try:
        with get_event_connection() as conn:
            rows = conn.execute(
                "PRAGMA table_info(security_alerts)"
            ).fetchall()
        return [row[1] for row in rows]
    except Exception:
        return []


def update_security_status(alert_id, new_status):
    if not table_exists("security_alerts"):
        return False, "security_alerts table does not exist."

    try:
        with get_event_connection() as conn:
            columns = set(security_table_columns())

            if (
                new_status == "ACKNOWLEDGED"
                and {"acknowledged_by", "acknowledged_at"}.issubset(columns)
            ):
                conn.execute(
                    """
                    UPDATE security_alerts
                    SET status = ?,
                        acknowledged_by = ?,
                        acknowledged_at = datetime('now')
                    WHERE id = ?
                    """,
                    (
                        new_status,
                        CURRENT_USER,
                        int(alert_id),
                    ),
                )
            elif (
                new_status == "RESOLVED"
                and "resolved_at" in columns
            ):
                conn.execute(
                    """
                    UPDATE security_alerts
                    SET status = ?,
                        resolved_at = datetime('now')
                    WHERE id = ?
                    """,
                    (
                        new_status,
                        int(alert_id),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE security_alerts
                    SET status = ?
                    WHERE id = ?
                    """,
                    (
                        new_status,
                        int(alert_id),
                    ),
                )

            conn.commit()

        write_audit(
            f"SECURITY_{new_status}",
            username=CURRENT_USER,
            role=CURRENT_ROLE,
            target=f"security_alerts:{alert_id}",
            details=f"Security alert status changed to {new_status}",
        )

        return True, None

    except Exception as exc:
        return False, str(exc)


# ============================================================
# IMAGE HELPERS
# ============================================================

def resolve_image_path(relative_path):
    if not relative_path:
        return None

    try:
        raw = Path(str(relative_path))

        candidates = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(ROOT / raw)

            # Some older records may contain a project-root path.
            candidates.append(raw)

        for path in candidates:
            if path.exists() and path.is_file():
                return path

    except (OSError, TypeError, ValueError):
        pass

    return None


def get_existing_evidence(events, limit=100):
    if events.empty or "image_path" not in events.columns:
        return pd.DataFrame()

    evidence = events[
        events["image_path"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    if evidence.empty:
        return evidence

    evidence = evidence.sort_values(
        ["id", "timestamp"],
        ascending=False,
    ).head(limit)

    evidence["resolved_path"] = evidence[
        "image_path"
    ].apply(resolve_image_path)

    return evidence[
        evidence["resolved_path"].notna()
    ].copy()


def get_latest_event_image():
    evidence = get_existing_evidence(
        load_events(),
        100,
    )

    if evidence.empty:
        return None

    row = evidence.iloc[0]

    return {
        "path": row["resolved_path"],
        "timestamp": row["timestamp"],
        "event_type": row["event_type"],
        "person_id": row["person_id"],
        "activity": row["activity"],
        "details": row["details"],
    }


# ============================================================
# ANALYTICS HELPERS
# ============================================================

def event_count(events, name):
    if events.empty:
        return 0

    return int(
        events["event_type"]
        .astype(str)
        .str.upper()
        .eq(name.upper())
        .sum()
    )


def count_alerts(events):
    if events.empty:
        return 0

    return int(
        events["event_type"]
        .astype(str)
        .str.upper()
        .str.contains(
            r"FALL|ALERT",
            regex=True,
            na=False,
        )
        .sum()
    )


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours}h {minutes}m"

    if minutes:
        return f"{minutes}m {secs}s"

    return f"{secs}s"


def get_latest_state(events):
    if events.empty:
        return pd.DataFrame()

    usable = events.dropna(
        subset=["person_id"]
    ).copy()

    if usable.empty:
        return pd.DataFrame()

    latest = (
        usable
        .sort_values(
            ["timestamp", "id"]
        )
        .groupby(
            "person_id",
            as_index=False,
        )
        .tail(1)
    )

    return latest.sort_values("person_id")


def posture_durations(events):
    columns = [
        "person_id",
        "activity",
        "start",
        "end",
        "duration_seconds",
    ]

    if events.empty:
        return pd.DataFrame(columns=columns)

    posture_events = events[
        events["event_type"]
        .astype(str)
        .str.upper()
        .eq("POSTURE_CHANGE")
    ].copy()

    if posture_events.empty:
        return pd.DataFrame(columns=columns)

    posture_events = posture_events.sort_values(
        ["person_id", "timestamp", "id"]
    )

    rows = []

    for person_id, group in posture_events.groupby(
        "person_id",
        dropna=False,
    ):
        group = group.reset_index(drop=True)

        for index in range(len(group) - 1):
            current = group.iloc[index]
            following = group.iloc[index + 1]

            if (
                pd.isna(current["timestamp"])
                or pd.isna(following["timestamp"])
            ):
                continue

            duration = (
                following["timestamp"]
                - current["timestamp"]
            ).total_seconds()

            if duration <= 0 or duration > 86400:
                continue

            rows.append(
                {
                    "person_id": person_id,
                    "activity": str(
                        current["activity"] or "UNKNOWN"
                    ).upper(),
                    "start": current["timestamp"],
                    "end": following["timestamp"],
                    "duration_seconds": duration,
                }
            )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def filter_by_date(events, selected_dates):
    if events.empty or not selected_dates:
        return events.copy()

    if (
        isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
    ):
        start_date, end_date = selected_dates

        return events[
            events["timestamp"].dt.date.ge(start_date)
            & events["timestamp"].dt.date.le(end_date)
        ].copy()

    return events.copy()


def security_summary(alerts):
    if alerts.empty:
        return {
            "total": 0,
            "open": 0,
            "critical": 0,
            "high": 0,
            "resolved": 0,
        }

    status = (
        alerts["status"]
        if "status" in alerts.columns
        else pd.Series("", index=alerts.index)
    )

    severity = (
        alerts["severity"]
        if "severity" in alerts.columns
        else pd.Series("", index=alerts.index)
    )

    status = (
        status.fillna("")
        .astype(str)
        .str.upper()
    )

    severity = (
        severity.fillna("")
        .astype(str)
        .str.upper()
    )

    return {
        "total": len(alerts),
        "open": int(
            status.isin(
                ["OPEN", "ACKNOWLEDGED"]
            ).sum()
        ),
        "critical": int(
            severity.eq("CRITICAL").sum()
        ),
        "high": int(
            severity.eq("HIGH").sum()
        ),
        "resolved": int(
            status.eq("RESOLVED").sum()
        ),
    }


def severity_class(severity):
    value = str(
        severity or ""
    ).upper()

    if value == "CRITICAL":
        return "security-critical"

    if value == "HIGH":
        return "security-high"

    if value == "MEDIUM":
        return "security-medium"

    return "security-normal"


# ============================================================
# LOAD
# ============================================================

df = load_events()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
    <div class="sidebar-brand-mark">📹</div>
    <div>
        <div class="sidebar-brand-name">AI Camera Tracker</div>
        <div class="sidebar-brand-sub">Command Center</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Theme Switcher Option
    selected_theme = st.radio(
        "Display Theme",
        options=["Dark", "Light"],
        horizontal=True,
        index=0 if st.session_state.theme == "Dark" else 1,
        help="Select between Dark Cyber Mode and High-Contrast Light Mode",
    )

    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.markdown("")

    role_badge = "🛡️" if IS_ADMIN else "👁️"

    st.markdown(
        f"""
<div class="sidebar-meta-card">
    <span style="font-weight:700; text-transform:uppercase; font-size:10px;">OPERATOR</span><br>
    <b>{CURRENT_USER}</b><br><br>
    <span style="font-weight:700; text-transform:uppercase; font-size:10px;">CLEARANCE</span><br>
    <b style="color:var(--neon-cyan);">{role_badge} {CURRENT_ROLE.upper()}</b><br><br>
    <span style="font-weight:700; text-transform:uppercase; font-size:10px;">DETECTION ENGINE</span><br>
    <b>🧠 DeepCamera / YOLO26</b>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    if IS_ADMIN:
        st.caption("Administrative access enabled.")
    else:
        st.caption("Viewer access enabled.")

    if st.button(
        "🔄 Refresh Dashboard",
        use_container_width=True,
    ):
        write_audit(
            "DASHBOARD_REFRESH",
            username=CURRENT_USER,
            role=CURRENT_ROLE,
        )
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Filters")

    selected_dates = None

    valid_dates = (
        df["timestamp"].dropna()
        if not df.empty
        else pd.Series(dtype="datetime64[ns]")
    )

    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_dates = st.date_input(
            "Monitoring period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    filtered = filter_by_date(
        df,
        selected_dates,
    )

    event_types = sorted(
        filtered["event_type"]
        .dropna()
        .astype(str)
        .unique()
    ) if not filtered.empty else []

    selected_event_types = st.multiselect(
        "Event types",
        options=event_types,
        default=event_types,
    )

    if selected_event_types:
        filtered = filtered[
            filtered["event_type"].isin(
                selected_event_types
            )
        ].copy()
    else:
        filtered = filtered.iloc[0:0].copy()

    st.markdown("---")

    st.markdown(
        """
<div class="sidebar-meta-card">
    <span style="font-weight:700; text-transform:uppercase; font-size:10px;">DATA INGESTION</span><br>
    <span>SQLite Local Event Logs</span><br><br>
    <span style="font-weight:700; text-transform:uppercase; font-size:10px;">PIPELINE CHAIN</span><br>
    <span>YOLO26 → Pose → Track → Classify → Security HUD</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button(
        "Logout",
        use_container_width=True,
    ):
        write_audit(
            "LOGOUT",
            username=CURRENT_USER,
            role=CURRENT_ROLE,
            details="User logged out",
        )

        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None

        st.rerun()


# ============================================================
# EMPTY DATABASE
# ============================================================

if df.empty:
    st.markdown(
        """
<div class="app-header">
    <div class="app-header-mark">📹</div>
    <div>
        <div class="app-title">AI Camera Tracker</div>
        <div class="app-subtitle">
            Intelligent monitoring, posture analytics,
            event intelligence and evidence management
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "No monitoring events are available yet. "
        "Start main.py and generate some events."
    )

    if IS_ADMIN:
        st.info(
            "Administrator access is active. "
            "You can still open Security Center and Administration "
            "from the navigation when event data becomes available."
        )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1.6],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        """
<div class="app-header">
    <div class="app-header-mark">📹</div>
    <div>
        <div class="app-title">AI Camera Tracker</div>
        <div class="app-subtitle">
            Intelligent monitoring, posture analytics,
            event intelligence and evidence management
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="display:flex; justify-content:flex-end; align-items:center;">'
        '<span class="status-online">Monitoring System Online</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("<div style='text-align:right;'>DeepCamera AI layer • YOLO26</div>", unsafe_allow_html=True)


# ============================================================
# ACCESS / SECURITY STATUS
# ============================================================

if IS_ADMIN:
    st.markdown(
        """
<div class="alert-banner alert-info">
    🔐 <b>Administrator session active.</b>
    Security Center includes user management, audit history,
    database visibility, security alerts and evidence review.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# KPI
# ============================================================

total_events = len(filtered)
people = filtered["person_id"].dropna().nunique()
entries = event_count(filtered, "ZONE_ENTRY")
limited = event_count(filtered, "LIMITED_VIEW")
falls = count_alerts(filtered)
posture_changes = event_count(filtered, "POSTURE_CHANGE")

kpi_items = [
    ("Total Events", total_events, "var(--neon-cyan)"),
    ("People", people, "var(--neon-blue)"),
    ("Zone Entries", entries, "#0284c7" if not is_dark else "#38bdf8"),
    ("Limited View", limited, "var(--neon-amber)"),
    ("Fall / Alerts", falls, "var(--neon-red)" if falls else "var(--neon-green)"),
    ("Posture Changes", posture_changes, "#9333ea" if not is_dark else "#a855f7"),
]

kpi_html = '<div class="kpi-row">'
for label, value, bar_color in kpi_items:
    kpi_html += (
        f'<div class="kpi-card" style="--accent-bar:{bar_color};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )
kpi_html += "</div>"

st.markdown(kpi_html, unsafe_allow_html=True)


# ============================================================
# ALERT SUMMARY
# ============================================================

if falls > 0:
    st.markdown(
        f"""
<div class="alert-banner alert-danger">
    🚨 <b>{falls} fall / alert event(s) detected</b> in the selected
    monitoring period. Review Events &amp; Evidence immediately.
</div>
""",
        unsafe_allow_html=True,
    )
elif limited > 0:
    st.markdown(
        f"""
<div class="alert-banner alert-warning">
    ⚠️ <b>{limited} limited-view event(s) detected.</b> Lower-body
    visibility may have affected posture classification.
</div>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="alert-banner alert-success">
    ✓ <b>Optimal telemetry.</b> No fall or critical alert events detected in the selected
    monitoring period.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# NAVIGATION
# ============================================================

(
    overview_tab,
    live_tab,
    posture_tab,
    reports_tab,
    events_tab,
    security_tab,
) = st.tabs(
    [
        "📊 Overview",
        "📡 Live Monitor",
        "🧍 Posture Analytics",
        "📅 Reports",
        "📝 Events & Evidence",
        "🔐 Security Center",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:
    st.markdown(
        '<div class="section-title">Monitoring Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Operational view of the camera tracking pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        st.markdown("#### Event Distribution")

        counts = (
            filtered["event_type"]
            .value_counts()
            .rename_axis("Event")
            .to_frame("Count")
        )

        if counts.empty:
            st.info("No event data available.")
        else:
            st.bar_chart(
                counts,
                use_container_width=True,
                color=chart_c1,
            )

    with right:
        st.markdown("#### Activity Distribution")

        activity_counts = (
            filtered["activity"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .value_counts()
            .rename_axis("Activity")
            .to_frame("Count")
        )

        if activity_counts.empty:
            st.info("No activity data available.")
        else:
            st.bar_chart(
                activity_counts,
                use_container_width=True,
                color=chart_c2,
            )

    st.markdown("---")
    st.markdown("#### Event Timeline")

    timeline = filtered.dropna(
        subset=["timestamp"]
    ).copy()

    if timeline.empty:
        st.info("No timeline data available.")
    else:
        timeline_counts = (
            timeline
            .set_index("timestamp")
            .resample("1min")
            .size()
        )

        st.area_chart(
            timeline_counts,
            use_container_width=True,
            color=chart_c3,
        )

    st.markdown("---")
    st.markdown("#### Latest Activity")

    latest = (
        filtered
        .sort_values(
            ["timestamp", "id"],
            ascending=False,
        )
        .head(10)
        .copy()
    )

    if latest.empty:
        st.info("No recent activity.")
    else:
        latest["timestamp"] = latest[
            "timestamp"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        latest_display = latest[
            [
                "timestamp",
                "person_id",
                "event_type",
                "activity",
                "details",
            ]
        ].rename(
            columns={
                "timestamp": "Timestamp",
                "person_id": "Person ID",
                "event_type": "Event",
                "activity": "Activity",
                "details": "Details",
            }
        )

        st.dataframe(
            latest_display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# LIVE MONITOR
# ============================================================

with live_tab:
    st.markdown(
        '<div class="section-title">Live Monitoring</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Latest captured state from the active camera pipeline. '
        'This dashboard does not open a second webcam stream.'
        '</div>',
        unsafe_allow_html=True,
    )

    latest_image = get_latest_event_image()

    if latest_image:
        left, right = st.columns(
            [2.2, 1],
            gap="large",
        )

        with left:
            st.image(
                str(latest_image["path"]),
                caption=(
                    f"{latest_image['event_type']} • "
                    f"Person {latest_image['person_id']}"
                ),
                use_container_width=True,
            )

        with right:
            st.markdown("#### Latest Detection")

            st.metric(
                "Event",
                str(latest_image["event_type"]),
            )

            st.write(
                f"**Person:** {latest_image['person_id']}"
            )

            st.write(
                f"**Activity:** "
                f"{latest_image['activity'] or 'UNKNOWN'}"
            )

            st.write(
                f"**Timestamp:** "
                f"{latest_image['timestamp']}"
            )

            if latest_image["details"]:
                st.write(
                    f"**Details:** {latest_image['details']}"
                )

            try:
                relative = latest_image["path"].relative_to(
                    ROOT
                )
                st.caption(
                    f"Evidence: {relative}"
                )
            except ValueError:
                st.caption(
                    str(latest_image["path"])
                )

    else:
        st.info(
            "No existing captured evidence image is available. "
            "Database records whose image files were deleted are "
            "ignored safely."
        )

    st.markdown("---")
    st.markdown("#### Current Person State")

    latest_state = get_latest_state(filtered)

    if latest_state.empty:
        st.info(
            "No person state is available for the selected filters."
        )
    else:
        state_display = latest_state[
            [
                "person_id",
                "timestamp",
                "event_type",
                "activity",
                "details",
            ]
        ].copy()

        state_display["timestamp"] = (
            state_display["timestamp"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        state_display = state_display.rename(
            columns={
                "person_id": "Person ID",
                "timestamp": "Last Update",
                "event_type": "Event",
                "activity": "Activity",
                "details": "Details",
            }
        )

        st.dataframe(
            state_display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# POSTURE ANALYTICS
# ============================================================

with posture_tab:
    st.markdown(
        '<div class="section-title">Posture Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Estimated posture duration from logged POSTURE_CHANGE events.'
        '</div>',
        unsafe_allow_html=True,
    )

    durations = posture_durations(df)

    if selected_dates and not durations.empty:
        if (
            isinstance(selected_dates, (tuple, list))
            and len(selected_dates) == 2
        ):
            durations = durations[
                durations["start"].dt.date.ge(
                    selected_dates[0]
                )
                & durations["start"].dt.date.le(
                    selected_dates[1]
                )
            ].copy()

    if durations.empty:
        st.info(
            "No POSTURE_CHANGE events are available yet. "
            "Posture duration analytics will appear here once "
            "posture changes are logged."
        )
    else:
        total_by_activity = (
            durations
            .groupby("activity")["duration_seconds"]
            .sum()
            .sort_values(ascending=False)
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Sitting",
            format_duration(
                total_by_activity.get(
                    "SITTING",
                    0,
                )
            ),
        )

        p2.metric(
            "Standing",
            format_duration(
                total_by_activity.get(
                    "STANDING",
                    0,
                )
            ),
        )

        p3.metric(
            "Unknown",
            format_duration(
                total_by_activity.get(
                    "UNKNOWN",
                    0,
                )
            ),
        )

        st.markdown("---")
        st.markdown("#### Time by Posture")

        chart_data = total_by_activity.rename(
            "seconds"
        ).to_frame()

        if chart_data.empty:
            st.info("No posture duration data.")
        else:
            st.bar_chart(
                chart_data,
                use_container_width=True,
                color=chart_c1,
            )

        st.markdown("---")
        st.markdown("#### Posture Intervals")

        posture_table = durations.copy()

        posture_table["Duration"] = posture_table[
            "duration_seconds"
        ].apply(format_duration)

        posture_table["Start"] = posture_table[
            "start"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        posture_table["End"] = posture_table[
            "end"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        posture_table = posture_table[
            [
                "person_id",
                "activity",
                "Start",
                "End",
                "Duration",
            ]
        ].sort_values(
            "Start",
            ascending=False,
        )

        posture_table = posture_table.rename(
            columns={
                "person_id": "Person ID",
                "activity": "Posture",
            }
        )

        st.dataframe(
            posture_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# REPORTS
# ============================================================

with reports_tab:
    st.markdown(
        '<div class="section-title">Monitoring Reports</div>',
        unsafe_allow_html=True,
    )

    report_df = filtered.dropna(
        subset=["timestamp"]
    ).copy()

    if report_df.empty:
        st.info("No report data available.")
    else:
        report_df["date"] = (
            report_df["timestamp"].dt.date
        )

        daily = (
            report_df
            .groupby("date")
            .agg(
                total_events=("id", "count"),
                people=("person_id", "nunique"),
            )
            .reset_index()
        )

        def daily_event_count(pattern):
            mask = report_df[
                "event_type"
            ].str.contains(
                pattern,
                regex=True,
                case=False,
                na=False,
            )

            return (
                report_df.loc[mask]
                .groupby("date")
                .size()
                .reindex(
                    daily["date"],
                    fill_value=0,
                )
                .values
            )

        daily["zone_entries"] = daily_event_count(
            r"^ZONE_ENTRY$"
        )

        daily["zone_exits"] = daily_event_count(
            r"^ZONE_EXIT$"
        )

        daily["limited_view"] = daily_event_count(
            r"^LIMITED_VIEW$"
        )

        daily["falls_alerts"] = daily_event_count(
            r"FALL|ALERT"
        )

        daily = daily.sort_values(
            "date",
            ascending=False,
        )

        st.markdown("#### Daily Summary")

        st.dataframe(
            daily.rename(
                columns={
                    "date": "Date",
                    "total_events": "Total Events",
                    "people": "People",
                    "zone_entries": "Zone Entries",
                    "zone_exits": "Zone Exits",
                    "limited_view": "Limited View",
                    "falls_alerts": "Falls / Alerts",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("#### Weekly Event Trend")

        weekly = (
            report_df
            .set_index("timestamp")
            .resample("W")
            .size()
        )

        if weekly.empty:
            st.info("No weekly trend data.")
        else:
            st.line_chart(
                weekly,
                use_container_width=True,
                color=chart_c1,
            )

        st.markdown("---")
        st.markdown("#### Report Summary")

        total_days = int(
            daily["date"].nunique()
        )

        average_events = float(
            daily["total_events"].mean()
        )

        peak_day = (
            daily.loc[
                daily["total_events"].idxmax(),
                "date",
            ]
            if not daily.empty
            else "N/A"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Monitoring Days",
            total_days,
        )

        r2.metric(
            "Avg Events / Day",
            f"{average_events:.1f}",
        )

        r3.metric(
            "Most Active Day",
            str(peak_day),
        )


# ============================================================
# EVENTS & EVIDENCE
# ============================================================

with events_tab:
    st.markdown(
        '<div class="section-title">Events & Evidence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Complete event history with safe handling of deleted evidence files.'
        '</div>',
        unsafe_allow_html=True,
    )

    display = filtered.copy()

    if display.empty:
        st.info("No events match the selected filters.")
    else:
        display["timestamp"] = display[
            "timestamp"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        display = display.rename(
            columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "person_id": "Person ID",
                "event_type": "Event",
                "activity": "Activity",
                "details": "Details",
                "image_path": "Evidence Path",
            }
        )

        st.dataframe(
            display[
                [
                    "ID",
                    "Timestamp",
                    "Person ID",
                    "Event",
                    "Activity",
                    "Details",
                    "Evidence Path",
                ]
            ].sort_values(
                "ID",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### Captured Evidence")

    evidence = get_existing_evidence(
        filtered,
        100,
    )

    if evidence.empty:
        st.info(
            "No existing evidence images are available for the "
            "selected filters. Deleted/missing images are ignored."
        )
    else:
        evidence_ids = evidence["id"].tolist()

        def evidence_label(event_id):
            row = evidence[
                evidence["id"] == event_id
            ].iloc[0]

            if pd.isna(row["person_id"]):
                person = "N/A"
            else:
                person = str(
                    int(row["person_id"])
                )

            return (
                f"Event #{event_id} | "
                f"{row['event_type']} | "
                f"Person {person}"
            )

        selected_id = st.selectbox(
            "Select event",
            evidence_ids,
            format_func=evidence_label,
            key="event_evidence_selector",
        )

        selected = evidence[
            evidence["id"] == selected_id
        ].iloc[0]

        image_path = selected["resolved_path"]

        if image_path and image_path.exists():
            left, right = st.columns(
                [2.2, 1],
                gap="large",
            )

            with left:
                st.image(
                    str(image_path),
                    use_container_width=True,
                )

            with right:
                st.markdown("#### Event Details")

                st.write(
                    f"**Timestamp:** {selected['timestamp']}"
                )

                st.write(
                    f"**Event:** {selected['event_type']}"
                )

                st.write(
                    f"**Person:** {selected['person_id']}"
                )

                st.write(
                    f"**Activity:** "
                    f"{selected['activity'] or 'UNKNOWN'}"
                )

                st.write(
                    f"**Details:** "
                    f"{selected['details'] or 'N/A'}"
                )

                st.caption(
                    f"Evidence: {selected['image_path']}"
                )
        else:
            st.warning(
                "The selected evidence file no longer exists."
            )


# ============================================================
# SECURITY CENTER
# ============================================================

with security_tab:
    st.markdown(
        '<div class="section-title">🔐 Security Command Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Centralized administration for users, audit logs, '
        'database visibility, security alerts and evidence.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not IS_ADMIN:
        st.warning(
            "Administrator privileges are required to access "
            "Security Center."
        )

    else:
        security_df = load_security_alerts()
        summary = security_summary(
            security_df
        )

        s1, s2, s3, s4, s5 = st.columns(5)

        s1.metric(
            "Total Alerts",
            summary["total"],
        )
        s2.metric(
            "Open",
            summary["open"],
        )
        s3.metric(
            "Critical",
            summary["critical"],
        )
        s4.metric(
            "High",
            summary["high"],
        )
        s5.metric(
            "Resolved",
            summary["resolved"],
        )

        if summary["critical"] > 0:
            st.markdown(
                f"""
<div class="alert-banner alert-danger">
    🚨 <b>CRITICAL SECURITY ATTENTION REQUIRED</b><br>
    {summary["critical"]} critical security alert(s) require administrator review.
</div>
""",
                unsafe_allow_html=True,
            )
        elif summary["open"] > 0:
            st.markdown(
                f"""
<div class="alert-banner alert-warning">
    ⚠️ <b>{summary["open"]} security alert(s) require administrator review.</b>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="alert-banner alert-success">
    ✓ <b>All sectors secure.</b> No open security alerts.
</div>
""",
                unsafe_allow_html=True,
            )

        (
            alerts_tab,
            security_evidence_tab,
            users_tab,
            audit_tab,
            database_tab,
            administration_tab,
        ) = st.tabs(
            [
                "🚨 Alerts",
                "📸 Evidence",
                "👥 Users",
                "📋 Audit Log",
                "🗄️ Database",
                "⚙️ Administration",
            ]
        )

        # ----------------------------------------------------
        # SECURITY ALERTS
        # ----------------------------------------------------

        with alerts_tab:
            if not table_exists("security_alerts"):
                st.info(
                    "The security_alerts table has not been created yet. "
                    "Start the updated main.py once to initialize security logging."
                )
            elif security_df.empty:
                st.info(
                    "No security alerts have been recorded yet."
                )
            else:
                c1, c2 = st.columns(2)

                with c1:
                    status_options = [
                        "ALL",
                        "OPEN",
                        "ACKNOWLEDGED",
                        "RESOLVED",
                    ]

                    status_filter = st.selectbox(
                        "Status",
                        status_options,
                        key="security_status_filter",
                    )

                with c2:
                    severity_options = ["ALL"]

                    if "severity" in security_df.columns:
                        severity_options += sorted(
                            security_df["severity"]
                            .dropna()
                            .astype(str)
                            .str.upper()
                            .unique()
                            .tolist()
                        )

                    severity_filter = st.selectbox(
                        "Severity",
                        severity_options,
                        key="security_severity_filter",
                    )

                view_df = security_df.copy()

                if (
                    status_filter != "ALL"
                    and "status" in view_df.columns
                ):
                    view_df = view_df[
                        view_df["status"]
                        .fillna("")
                        .astype(str)
                        .str.upper()
                        .eq(status_filter)
                    ]

                if (
                    severity_filter != "ALL"
                    and "severity" in view_df.columns
                ):
                    view_df = view_df[
                        view_df["severity"]
                        .fillna("")
                        .astype(str)
                        .str.upper()
                        .eq(severity_filter)
                    ]

                if view_df.empty:
                    st.info(
                        "No security alerts match the selected filters."
                    )
                else:
                    preferred = [
                        "id",
                        "created_at",
                        "timestamp",
                        "person_id",
                        "severity",
                        "threat_type",
                        "status",
                        "notification_status",
                        "details",
                    ]

                    cols = [
                        col
                        for col in preferred
                        if col in view_df.columns
                    ]

                    display_security = view_df[
                        cols
                    ].copy()

                    display_security = display_security.rename(
                        columns={
                            "id": "Alert ID",
                            "created_at": "Created",
                            "timestamp": "Timestamp",
                            "person_id": "Person",
                            "severity": "Severity",
                            "threat_type": "Threat",
                            "status": "Status",
                            "notification_status": "Notification",
                            "details": "Details",
                        }
                    )

                    st.dataframe(
                        display_security,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("---")

                    selected_alert_id = st.selectbox(
                        "Review security alert",
                        view_df["id"].tolist(),
                        format_func=lambda value:
                            f"Security Alert #{value}",
                        key="security_alert_selector",
                    )

                    selected_alert = view_df[
                        view_df["id"] == selected_alert_id
                    ].iloc[0]

                    left, right = st.columns(
                        [1, 1],
                        gap="large",
                    )

                    with left:
                        st.markdown("#### Incident Details")

                        severity = selected_alert.get(
                            "severity",
                            "UNKNOWN",
                        )

                        threat = selected_alert.get(
                            "threat_type",
                            "UNKNOWN",
                        )

                        status = selected_alert.get(
                            "status",
                            "UNKNOWN",
                        )

                        person = selected_alert.get(
                            "person_id",
                            "N/A",
                        )

                        created = selected_alert.get(
                            "created_at",
                            selected_alert.get(
                                "timestamp",
                                "N/A",
                            ),
                        )

                        st.markdown(
                            f"""
<div class="security-card {severity_class(severity)}">
    <div style="font-size:24px;font-weight:800; font-family:'JetBrains Mono', monospace;">
        ALERT #{selected_alert_id}
    </div>
    <div style="color:var(--text-muted); font-size:12px; margin-top:2px;">
        Security Incident Record
    </div>
    <br>
    <b>Threat Vector:</b> {threat}<br>
    <b>Severity Level:</b> {severity}<br>
    <b>Person Subject:</b> {person}<br>
    <b>Current Status:</b> {status}<br>
    <b>Incident Logged:</b> {created}
</div>
""",
                            unsafe_allow_html=True,
                        )

                        details = selected_alert.get(
                            "details",
                            "",
                        )

                        if details:
                            st.info(str(details))

                        if "notification_status" in selected_alert.index:
                            st.write(
                                "**Notification:** "
                                + str(
                                    selected_alert[
                                        "notification_status"
                                    ]
                                )
                            )

                    with right:
                        st.markdown("#### Security Evidence")

                        evidence_path = selected_alert.get(
                            "evidence_path",
                            "",
                        )

                        if evidence_path:
                            resolved = resolve_image_path(
                                evidence_path
                            )

                            if resolved:
                                st.image(
                                    str(resolved),
                                    use_container_width=True,
                                    caption="Security incident evidence",
                                )
                            else:
                                st.warning(
                                    "The alert references an evidence "
                                    "image that no longer exists."
                                )
                                st.caption(
                                    f"Recorded path: {evidence_path}"
                                )
                        else:
                            st.info(
                                "No evidence path is associated with "
                                "this security alert."
                            )

                    st.markdown("#### Administrator Actions")

                    a1, a2 = st.columns(2)

                    current_status = str(
                        selected_alert.get(
                            "status",
                            "",
                        )
                    ).upper()

                    with a1:
                        if current_status == "OPEN":
                            if st.button(
                                "✓ Acknowledge Alert",
                                use_container_width=True,
                                key=f"ack_{selected_alert_id}",
                            ):
                                success, error = update_security_status(
                                    selected_alert_id,
                                    "ACKNOWLEDGED",
                                )

                                if success:
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error(error)
                        else:
                            st.button(
                                "✓ Acknowledged",
                                disabled=True,
                                use_container_width=True,
                                key=f"ack_disabled_{selected_alert_id}",
                            )

                    with a2:
                        if current_status != "RESOLVED":
                            if st.button(
                                "✓ Resolve Alert",
                                use_container_width=True,
                                key=f"resolve_{selected_alert_id}",
                            ):
                                success, error = update_security_status(
                                    selected_alert_id,
                                    "RESOLVED",
                                )

                                if success:
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error(error)
                        else:
                            st.button(
                                "✓ Resolved",
                                disabled=True,
                                use_container_width=True,
                                key=f"resolved_{selected_alert_id}",
                            )

        # ----------------------------------------------------
        # SECURITY EVIDENCE
        # ----------------------------------------------------

        with security_evidence_tab:
            if security_df.empty:
                st.info(
                    "No security evidence is available yet."
                )
            elif "evidence_path" not in security_df.columns:
                st.warning(
                    "The security database does not contain "
                    "an evidence_path column."
                )
            else:
                evidence_df = security_df[
                    security_df["evidence_path"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                ].copy()

                if evidence_df.empty:
                    st.info(
                        "No security alerts currently contain evidence paths."
                    )
                else:
                    evidence_df["resolved_path"] = (
                        evidence_df["evidence_path"].apply(
                            resolve_image_path
                        )
                    )

                    evidence_df = evidence_df[
                        evidence_df["resolved_path"].notna()
                    ].copy()

                    if evidence_df.empty:
                        st.warning(
                            "All referenced security evidence images "
                            "have been deleted or are unavailable."
                        )
                    else:
                        selected_evidence_id = st.selectbox(
                            "Select security evidence",
                            evidence_df["id"].tolist(),
                            format_func=lambda value:
                                f"Security Alert #{value}",
                            key="security_evidence_selector",
                        )

                        evidence_row = evidence_df[
                            evidence_df["id"]
                            == selected_evidence_id
                        ].iloc[0]

                        left, right = st.columns(
                            [2.2, 1],
                            gap="large",
                        )

                        with left:
                            st.image(
                                str(
                                    evidence_row[
                                        "resolved_path"
                                    ]
                                ),
                                use_container_width=True,
                                caption="Security evidence",
                            )

                        with right:
                            st.markdown("#### Incident")

                            st.write(
                                f"**Alert:** "
                                f"#{evidence_row['id']}"
                            )

                            st.write(
                                f"**Threat:** "
                                f"{evidence_row.get('threat_type', 'UNKNOWN')}"
                            )

                            st.write(
                                f"**Severity:** "
                                f"{evidence_row.get('severity', 'UNKNOWN')}"
                            )

                            st.write(
                                f"**Person:** "
                                f"{evidence_row.get('person_id', 'N/A')}"
                            )

                            st.write(
                                f"**Status:** "
                                f"{evidence_row.get('status', 'UNKNOWN')}"
                            )

        # ----------------------------------------------------
        # USERS
        # ----------------------------------------------------

        with users_tab:
            st.markdown("#### User Management")

            users_df = load_users()

            if not users_df.empty:
                st.dataframe(
                    users_df.rename(
                        columns={
                            "id": "ID",
                            "username": "Username",
                            "role": "Role",
                            "created_at": "Created",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("---")
            st.markdown("##### Create User")

            with st.form("create_user_form"):
                new_username = st.text_input(
                    "Username",
                    key="admin_new_username",
                )

                new_password = st.text_input(
                    "Temporary password",
                    type="password",
                    key="admin_new_password",
                )

                new_role = st.selectbox(
                    "Role",
                    ["viewer", "admin"],
                    key="admin_new_role",
                )

                create_submitted = st.form_submit_button(
                    "Create User",
                    use_container_width=True,
                )

            if create_submitted:
                clean_new_username = new_username.strip()

                if not clean_new_username:
                    st.error(
                        "Username cannot be empty."
                    )
                elif len(new_password) < 6:
                    st.error(
                        "Password must contain at least 6 characters."
                    )
                else:
                    try:
                        create_user(
                            clean_new_username,
                            new_password,
                            new_role,
                        )

                        write_audit(
                            "USER_CREATED",
                            username=CURRENT_USER,
                            role=CURRENT_ROLE,
                            target=clean_new_username,
                            details=f"Created {new_role} user",
                        )

                        st.success(
                            f"User '{clean_new_username}' created."
                        )
                        st.rerun()

                    except Exception as exc:
                        st.error(
                            f"Could not create user: {exc}"
                        )

            st.markdown("---")
            st.markdown("##### Remove User")

            removable = []

            if not users_df.empty:
                removable = [
                    username
                    for username in users_df["username"].tolist()
                    if username != CURRENT_USER
                ]

            if removable:
                user_to_remove = st.selectbox(
                    "Select user",
                    removable,
                    key="remove_user_selector",
                )

                if st.button(
                    "Remove Selected User",
                    type="secondary",
                    use_container_width=True,
                ):
                    success, error = delete_user(
                        user_to_remove
                    )

                    if success:
                        write_audit(
                            "USER_REMOVED",
                            username=CURRENT_USER,
                            role=CURRENT_ROLE,
                            target=user_to_remove,
                            details="User account removed",
                        )

                        st.success(
                            f"User '{user_to_remove}' removed."
                        )
                        st.rerun()
                    else:
                        st.error(error)
            else:
                st.info(
                    "No other user accounts are available to remove."
                )

        # ----------------------------------------------------
        # AUDIT LOG
        # ----------------------------------------------------

        with audit_tab:
            st.markdown("#### Administrative Audit Trail")

            audit_df = load_audit_logs()

            if audit_df.empty:
                st.info(
                    "No audit records have been created yet."
                )
            else:
                st.dataframe(
                    audit_df.rename(
                        columns={
                            "id": "ID",
                            "timestamp": "Timestamp",
                            "username": "Username",
                            "role": "Role",
                            "action": "Action",
                            "target": "Target",
                            "details": "Details",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        with database_tab:
            st.markdown("#### Database Administration")

            st.write(
                "Read-only operational visibility into the SQLite "
                "databases used by authentication and camera monitoring."
            )

            d1, d2 = st.columns(2)

            with d1:
                if EVENT_DB_PATH.exists():
                    st.success(
                        "🟢 Camera event database available"
                    )

                    try:
                        size_mb = (
                            EVENT_DB_PATH.stat().st_size
                            / (1024 * 1024)
                        )
                        st.metric(
                            "Event DB Size",
                            f"{size_mb:.2f} MB",
                        )
                    except OSError:
                        pass

                    st.caption(
                        str(EVENT_DB_PATH)
                    )
                else:
                    st.error(
                        "🔴 Camera event database unavailable"
                    )

            with d2:
                if AUTH_DB_PATH.exists():
                    st.success(
                        "🟢 Authentication database available"
                    )

                    try:
                        size_mb = (
                            AUTH_DB_PATH.stat().st_size
                            / (1024 * 1024)
                        )
                        st.metric(
                            "Auth DB Size",
                            f"{size_mb:.2f} MB",
                        )
                    except OSError:
                        pass

                    st.caption(
                        str(AUTH_DB_PATH)
                    )
                else:
                    st.error(
                        "🔴 Authentication database unavailable"
                    )

            st.markdown("---")
            st.markdown("##### Camera Database Tables")

            tables = event_db_tables()

            if tables:
                st.dataframe(
                    pd.DataFrame(
                        {"Table": tables}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No camera database tables available."
                )

            st.markdown("---")
            st.markdown("##### Event Database Overview")

            if EVENT_DB_PATH.exists():
                try:
                    with get_event_connection() as conn:
                        event_count_db = conn.execute(
                            "SELECT COUNT(*) FROM events"
                        ).fetchone()[0]

                    st.metric(
                        "Rows in events table",
                        int(event_count_db),
                    )

                    if table_exists("security_alerts"):
                        with get_event_connection() as conn:
                            alert_count_db = conn.execute(
                                "SELECT COUNT(*) FROM security_alerts"
                            ).fetchone()[0]

                        st.metric(
                            "Rows in security_alerts",
                            int(alert_count_db),
                        )
                except Exception as exc:
                    st.warning(
                        f"Could not inspect database: {exc}"
                    )

            st.markdown("---")
            st.markdown("##### Authentication Database")

            auth_users = load_users()

            if auth_users.empty:
                st.info(
                    "No users found."
                )
            else:
                st.dataframe(
                    auth_users.rename(
                        columns={
                            "id": "ID",
                            "username": "Username",
                            "role": "Role",
                            "created_at": "Created",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            write_audit(
                "DATABASE_VIEW",
                username=CURRENT_USER,
                role=CURRENT_ROLE,
                target="camera_events.db / users.db",
                details="Administrator opened database administration",
            )

        # ----------------------------------------------------
        # ADMINISTRATION
        # ----------------------------------------------------

        with administration_tab:
            st.markdown("#### Security & Administration")

            st.write(
                "Manage your account and review the operational "
                "security configuration."
            )

            st.markdown("---")
            st.markdown("##### 📷 Camera Control")

            st.caption(
                "Administrator controls for camera permission and "
                "automatic camera triggering."
            )

            camera_state = load_camera_control()

            current_allowed = bool(
                camera_state.get(
                    "camera_allowed",
                    True,
                )
            )

            current_auto_start = bool(
                camera_state.get(
                    "auto_start",
                    True,
                )
            )

            c1, c2 = st.columns(2)

            with c1:
                allow_camera = st.toggle(
                    "Allow Camera Access",
                    value=current_allowed,
                    key="admin_camera_allowed",
                    help=(
                        "OFF releases the webcam and prevents "
                        "the camera process from processing frames."
                    ),
                )

            with c2:
                auto_trigger = st.toggle(
                    "Automatic Camera Trigger",
                    value=current_auto_start,
                    key="admin_camera_auto_start",
                    help=(
                        "Automatically starts/reconnects the camera "
                        "when camera access is allowed."
                    ),
                )

            if allow_camera != current_allowed:
                camera_state["camera_allowed"] = allow_camera

                save_camera_control(
                    camera_state,
                    updated_by=CURRENT_USER,
                )

                write_audit(
                    "CAMERA_ACCESS_CHANGED",
                    username=CURRENT_USER,
                    role=CURRENT_ROLE,
                    target="camera_control",
                    details=(
                        "camera_allowed="
                        f"{allow_camera}"
                    ),
                )

                st.rerun()

            if auto_trigger != current_auto_start:
                camera_state["auto_start"] = auto_trigger

                save_camera_control(
                    camera_state,
                    updated_by=CURRENT_USER,
                )

                write_audit(
                    "CAMERA_AUTO_TRIGGER_CHANGED",
                    username=CURRENT_USER,
                    role=CURRENT_ROLE,
                    target="camera_control",
                    details=(
                        "auto_start="
                        f"{auto_trigger}"
                    ),
                )

                st.rerun()

            latest_camera_state = load_camera_control()

            if latest_camera_state.get(
                "camera_allowed",
                True,
            ):
                st.success(
                    "🟢 Camera access is ALLOWED"
                )
            else:
                st.error(
                    "🔴 Camera access is BLOCKED"
                )

            if latest_camera_state.get(
                "auto_start",
                True,
            ):
                st.info(
                    "⚡ Automatic camera trigger is ENABLED."
                )
            else:
                st.warning(
                    "⏸️ Automatic camera trigger is DISABLED."
                )

            st.caption(
                "Last changed by: "
                + str(
                    latest_camera_state.get(
                        "updated_by",
                        "system",
                    )
                )
            )

            st.markdown("---")
            st.markdown("##### Change Your Password")

            with st.form("change_password_form"):
                new_password = st.text_input(
                    "New password",
                    type="password",
                    key="change_password_new",
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    key="change_password_confirm",
                )

                password_submitted = st.form_submit_button(
                    "Change Password",
                    use_container_width=True,
                )

            if password_submitted:
                if len(new_password) < 6:
                    st.error(
                        "Password must contain at least 6 characters."
                    )
                elif new_password != confirm_password:
                    st.error(
                        "Passwords do not match."
                    )
                else:
                    try:
                        change_password(
                            CURRENT_USER,
                            new_password,
                        )

                        write_audit(
                            "PASSWORD_CHANGED",
                            username=CURRENT_USER,
                            role=CURRENT_ROLE,
                            target=CURRENT_USER,
                            details="Password changed",
                        )

                        st.success(
                            "Password changed successfully."
                        )
                    except Exception as exc:
                        st.error(
                            f"Could not change password: {exc}"
                        )

            st.markdown("---")
            st.markdown("##### Notification Channels")

            telegram_configured = bool(
                os.getenv("TELEGRAM_BOT_TOKEN")
                and os.getenv("TELEGRAM_CHAT_ID")
            )

            email_configured = bool(
                os.getenv("SMTP_HOST")
                and os.getenv("SMTP_USERNAME")
                and os.getenv("SMTP_PASSWORD")
                and os.getenv("ADMIN_EMAIL")
            )

            n1, n2 = st.columns(2)

            with n1:
                if telegram_configured:
                    st.success(
                        "🟢 Telegram configured"
                    )
                else:
                    st.warning(
                        "⚪ Telegram not configured"
                    )

            with n2:
                if email_configured:
                    st.success(
                        "🟢 Email configured"
                    )
                else:
                    st.warning(
                        "⚪ Email not configured"
                    )

            st.caption(
                "Notification credentials should be stored in "
                "Streamlit Secrets or environment variables. "
                "Never commit tokens or passwords to GitHub."
            )

            st.markdown("---")
            st.markdown("##### Security Monitoring Scope")

            scope_df = pd.DataFrame(
                [
                    {
                        "Detection": "Threat Object",
                        "Severity": "CRITICAL",
                        "Admin Response": "Notify + Evidence",
                    },
                    {
                        "Detection": "Confirmed Fall",
                        "Severity": "CRITICAL",
                        "Admin Response": "Notify + Evidence",
                    },
                    {
                        "Detection": "Restricted Zone",
                        "Severity": "HIGH",
                        "Admin Response": "Notify + Evidence",
                    },
                    {
                        "Detection": "Limited View",
                        "Severity": "MONITORING",
                        "Admin Response": "Normal Event Log",
                    },
                ]
            )

            st.dataframe(
                scope_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.markdown("##### Current Session")

            session_df = pd.DataFrame(
                [
                    {
                        "Username": CURRENT_USER,
                        "Role": CURRENT_ROLE.upper(),
                        "Authentication": "ACTIVE",
                        "Audit Logging": "ENABLED",
                    }
                ]
            )

            st.dataframe(
                session_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    AI Camera Tracker • Command Center •
    SQLite Event Analytics • Role-Based Administration • Audit Logging
</div>
""",
    unsafe_allow_html=True,
)