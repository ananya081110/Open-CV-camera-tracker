
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "logs" / "camera_events.db"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Camera Tracker | Command Center",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UI
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background: #f5f7fb;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Force readable text regardless of Streamlit theme */
html, body, [class*="css"] {
    color: #111827;
}

h1, h2, h3, h4, h5, h6,
p, label, span, div {
    color: #111827;
}

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #e5e7eb !important;
}

section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    border-radius: 8px;
    border: 1px solid #374151;
    background: #1f2937;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stButton button:hover {
    border-color: #60a5fa;
    background: #263449;
}

.app-header {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.app-title {
    font-size: 30px;
    font-weight: 750;
    color: #111827 !important;
    letter-spacing: -0.025em;
}

.app-subtitle {
    margin-top: 4px;
    color: #6b7280 !important;
    font-size: 14px;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #111827 !important;
    margin-top: 8px;
    margin-bottom: 3px;
}

.section-subtitle {
    color: #6b7280 !important;
    font-size: 13px;
    margin-bottom: 14px;
}

.status-online {
    display: inline-block;
    background: #ecfdf5;
    color: #047857 !important;
    border: 1px solid #a7f3d0;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 650;
}

.alert-banner {
    border-radius: 10px;
    padding: 13px 16px;
    margin: 12px 0 18px 0;
    font-size: 13px;
    font-weight: 650;
}

.alert-danger {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b !important;
}

.alert-warning {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e !important;
}

.alert-success {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46 !important;
}

.alert-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af !important;
}

.security-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 7px rgba(15, 23, 42, 0.035);
}

.security-critical {
    border-left: 5px solid #dc2626;
}

.security-high {
    border-left: 5px solid #ea580c;
}

.security-medium {
    border-left: 5px solid #d97706;
}

.security-normal {
    border-left: 5px solid #16a34a;
}

.security-number {
    font-size: 27px;
    font-weight: 750;
    color: #111827 !important;
}

.security-label {
    color: #6b7280 !important;
    font-size: 12px;
}

.footer {
    color: #9ca3af !important;
    font-size: 11px;
    text-align: center;
    padding-top: 28px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stMarkdownContainer"] {
    color: #111827;
}

button[data-baseweb="tab"] {
    font-weight: 650;
    font-size: 13px;
    color: #374151 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


@st.cache_data(ttl=3)
def load_events():
    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        with get_connection() as conn:
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

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")
    df["event_type"] = (
        df["event_type"].fillna("UNKNOWN").astype(str).str.upper()
    )
    df["activity"] = (
        df["activity"].fillna("UNKNOWN").astype(str).str.upper()
    )
    df["details"] = df["details"].fillna("").astype(str)
    df["image_path"] = df["image_path"].fillna("").astype(str)

    return df


@st.cache_data(ttl=3)
def security_table_exists():
    if not DB_PATH.exists():
        return False

    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                AND name = 'security_alerts'
                """
            ).fetchone()
        return row is not None
    except Exception:
        return False


@st.cache_data(ttl=3)
def load_security_alerts():
    if not security_table_exists():
        return pd.DataFrame()

    try:
        with get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM security_alerts ORDER BY id DESC",
                conn,
            )
    except Exception as exc:
        st.error(f"Could not read security alerts: {exc}")
        return pd.DataFrame()


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

            # Also support paths accidentally stored with the
            # project root prefix.
            if str(raw).startswith(str(ROOT)):
                candidates.append(raw)

        for path in candidates:
            if path.exists() and path.is_file():
                return path

    except (OSError, TypeError, ValueError):
        pass

    return None


def get_existing_evidence(events, limit=100):
    if events.empty:
        return pd.DataFrame()

    evidence = events[
        events["image_path"].astype(str).str.strip().ne("")
    ].copy()

    if evidence.empty:
        return evidence

    evidence = evidence.sort_values(
        ["id", "timestamp"],
        ascending=False,
    ).head(limit)

    evidence["resolved_path"] = evidence["image_path"].apply(
        resolve_image_path
    )

    return evidence[evidence["resolved_path"].notna()].copy()


def get_latest_event_image():
    evidence = get_existing_evidence(load_events(), 100)

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
# ANALYTICS
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
        .str.contains(r"FALL|ALERT", regex=True, na=False)
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

    usable = events.dropna(subset=["person_id"]).copy()

    if usable.empty:
        return pd.DataFrame()

    latest = (
        usable.sort_values(["timestamp", "id"])
        .groupby("person_id", as_index=False)
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
        "person_id", dropna=False
    ):
        group = group.reset_index(drop=True)

        for index in range(len(group) - 1):
            current = group.iloc[index]
            following = group.iloc[index + 1]

            if pd.isna(current["timestamp"]) or pd.isna(
                following["timestamp"]
            ):
                continue

            duration = (
                following["timestamp"] - current["timestamp"]
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

    return pd.DataFrame(rows, columns=columns)


def filter_by_date(events, selected_dates):
    if events.empty or not selected_dates:
        return events.copy()

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        return events[
            events["timestamp"].dt.date.ge(start_date)
            & events["timestamp"].dt.date.le(end_date)
        ].copy()

    return events.copy()


# ============================================================
# SECURITY HELPERS
# ============================================================

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

    status = status.fillna("").astype(str).str.upper()
    severity = severity.fillna("").astype(str).str.upper()

    return {
        "total": len(alerts),
        "open": int(status.isin(["OPEN", "ACKNOWLEDGED"]).sum()),
        "critical": int(severity.eq("CRITICAL").sum()),
        "high": int(severity.eq("HIGH").sum()),
        "resolved": int(status.eq("RESOLVED").sum()),
    }


def update_security_status(alert_id, new_status):
    if not security_table_exists():
        return False, "security_alerts table does not exist."

    try:
        with get_connection() as conn:
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(security_alerts)"
                ).fetchall()
            }

            if new_status == "ACKNOWLEDGED" and {
                "acknowledged_by",
                "acknowledged_at",
            }.issubset(columns):
                conn.execute(
                    """
                    UPDATE security_alerts
                    SET status = ?,
                        acknowledged_by = ?,
                        acknowledged_at = datetime('now')
                    WHERE id = ?
                    """,
                    (new_status, "admin", int(alert_id)),
                )
            elif new_status == "RESOLVED" and "resolved_at" in columns:
                conn.execute(
                    """
                    UPDATE security_alerts
                    SET status = ?,
                        resolved_at = datetime('now')
                    WHERE id = ?
                    """,
                    (new_status, int(alert_id)),
                )
            else:
                conn.execute(
                    """
                    UPDATE security_alerts
                    SET status = ?
                    WHERE id = ?
                    """,
                    (new_status, int(alert_id)),
                )

            conn.commit()

        return True, None

    except Exception as exc:
        return False, str(exc)


def severity_class(severity):
    value = str(severity or "").upper()

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
<div style="font-size:20px;font-weight:750;color:#ffffff !important;">
    AI Camera Tracker
</div>
<div style="font-size:11px;color:#9ca3af !important;margin-top:3px;">
    Monitoring Command Center
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Filters")

    selected_dates = None

    valid_dates = df["timestamp"].dropna()

    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_dates = st.date_input(
            "Monitoring period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    filtered = filter_by_date(df, selected_dates)

    event_types = sorted(
        filtered["event_type"].dropna().astype(str).unique()
    )

    selected_event_types = st.multiselect(
        "Event types",
        options=event_types,
        default=event_types,
    )

    if selected_event_types:
        filtered = filtered[
            filtered["event_type"].isin(selected_event_types)
        ].copy()
    else:
        filtered = filtered.iloc[0:0].copy()

    st.markdown("---")

    st.markdown(
        """
<div style="font-size:11px;color:#9ca3af !important;line-height:1.7;">
    <b>Data source</b><br>
    SQLite event database<br><br>

    <b>Pipeline</b><br>
    Pose Detection → Tracking → Classification →
    Event Logging → Analytics → Security
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# EMPTY DATABASE
# ============================================================

if df.empty:
    st.markdown(
        """
<div class="app-header">
    <div class="app-title">📹 AI Camera Tracker</div>
    <div class="app-subtitle">
        Intelligent monitoring, posture analytics,
        event intelligence and evidence management
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "No monitoring events are available yet. "
        "Start main.py and generate some events."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        """
<div class="app-header">
    <div class="app-title">📹 AI Camera Tracker</div>
    <div class="app-subtitle">
        Intelligent monitoring, posture analytics,
        event intelligence and evidence management
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div class="status-online">● Monitoring System Online</div>',
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

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Total Events", total_events)
k2.metric("People", people)
k3.metric("Zone Entries", entries)
k4.metric("Limited View", limited)
k5.metric("Fall / Alerts", falls)
k6.metric("Posture Changes", posture_changes)


# ============================================================
# ALERT
# ============================================================

if falls > 0:
    st.markdown(
        f"""
<div class="alert-banner alert-danger">
    🚨 {falls} fall / alert event(s) detected in the selected
    monitoring period. Review Events &amp; Evidence immediately.
</div>
""",
        unsafe_allow_html=True,
    )
elif limited > 0:
    st.markdown(
        f"""
<div class="alert-banner alert-warning">
    ⚠️ {limited} limited-view event(s) detected. Lower-body
    visibility may have affected posture classification.
</div>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="alert-banner alert-success">
    ✓ No fall or critical alert events detected in the selected
    monitoring period.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# TABS
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

    left, right = st.columns(2, gap="large")

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
            st.bar_chart(counts, use_container_width=True)

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
            st.bar_chart(activity_counts, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Event Timeline")

    timeline = filtered.dropna(subset=["timestamp"]).copy()

    if timeline.empty:
        st.info("No timeline data available.")
    else:
        timeline_counts = (
            timeline.set_index("timestamp").resample("1min").size()
        )
        st.line_chart(timeline_counts, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Latest Activity")

    latest = (
        filtered.sort_values(
            ["timestamp", "id"],
            ascending=False,
        )
        .head(10)
        .copy()
    )

    if latest.empty:
        st.info("No recent activity.")
    else:
        latest["timestamp"] = latest["timestamp"].dt.strftime(
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
        left, right = st.columns([2.2, 1], gap="large")

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
            st.metric("Event", str(latest_image["event_type"]))
            st.write(f"**Person:** {latest_image['person_id']}")
            st.write(
                f"**Activity:** "
                f"{latest_image['activity'] or 'UNKNOWN'}"
            )
            st.write(
                f"**Timestamp:** {latest_image['timestamp']}"
            )

            if latest_image["details"]:
                st.write(
                    f"**Details:** {latest_image['details']}"
                )

            try:
                relative = latest_image["path"].relative_to(ROOT)
                st.caption(f"Evidence: {relative}")
            except ValueError:
                st.caption(str(latest_image["path"]))

    else:
        st.info(
            "No existing captured evidence image is available. "
            "The database may contain records whose image files "
            "were deleted."
        )

    st.markdown("---")
    st.markdown("#### Current Person State")

    latest_state = get_latest_state(filtered)

    if latest_state.empty:
        st.info("No person state is available for the selected filters.")
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
            state_display["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
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
# POSTURE
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
                durations["start"].dt.date.ge(selected_dates[0])
                & durations["start"].dt.date.le(selected_dates[1])
            ].copy()

    if durations.empty:
        st.info(
            "No POSTURE_CHANGE events are available yet. "
            "Posture duration analytics will appear here once "
            "posture changes are logged."
        )
    else:
        total_by_activity = (
            durations.groupby("activity")["duration_seconds"]
            .sum()
            .sort_values(ascending=False)
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Sitting",
            format_duration(
                total_by_activity.get("SITTING", 0)
            ),
        )

        p2.metric(
            "Standing",
            format_duration(
                total_by_activity.get("STANDING", 0)
            ),
        )

        p3.metric(
            "Unknown",
            format_duration(
                total_by_activity.get("UNKNOWN", 0)
            ),
        )

        st.markdown("---")
        st.markdown("#### Time by Posture")

        chart_data = total_by_activity.rename("seconds").to_frame()

        st.bar_chart(chart_data, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Posture Intervals")

        posture_table = durations.copy()

        posture_table["Duration"] = posture_table[
            "duration_seconds"
        ].apply(format_duration)

        posture_table["Start"] = posture_table["start"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        posture_table["End"] = posture_table["end"].dt.strftime(
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
        ].sort_values("Start", ascending=False)

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

    report_df = filtered.dropna(subset=["timestamp"]).copy()

    if report_df.empty:
        st.info("No report data available.")
    else:
        report_df["date"] = report_df["timestamp"].dt.date

        daily = (
            report_df.groupby("date")
            .agg(
                total_events=("id", "count"),
                people=("person_id", "nunique"),
            )
            .reset_index()
        )

        def daily_count(event_name):
            mask = (
                report_df["event_type"]
                .astype(str)
                .str.upper()
                .eq(event_name)
            )

            return (
                report_df.loc[mask]
                .groupby("date")
                .size()
                .reindex(daily["date"], fill_value=0)
                .values
            )

        daily["zone_entries"] = daily_count("ZONE_ENTRY")
        daily["zone_exits"] = daily_count("ZONE_EXIT")
        daily["limited_view"] = daily_count("LIMITED_VIEW")

        fall_mask = (
            report_df["event_type"]
            .astype(str)
            .str.upper()
            .str.contains("FALL|ALERT", regex=True, na=False)
        )

        daily["falls_alerts"] = (
            report_df.loc[fall_mask]
            .groupby("date")
            .size()
            .reindex(daily["date"], fill_value=0)
            .values
        )

        daily = daily.sort_values("date", ascending=False)

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
            report_df.set_index("timestamp")
            .resample("W")
            .size()
        )

        if not weekly.empty:
            st.line_chart(weekly, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Report Summary")

        total_days = int(daily["date"].nunique())
        average_events = float(daily["total_events"].mean())

        peak_day = (
            daily.loc[
                daily["total_events"].idxmax(),
                "date",
            ]
            if not daily.empty
            else "N/A"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric("Monitoring Days", total_days)
        r2.metric("Avg Events / Day", f"{average_events:.1f}")
        r3.metric("Most Active Day", str(peak_day))


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
        display["timestamp"] = display["timestamp"].dt.strftime(
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
            ].sort_values("ID", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### Captured Evidence")

    evidence = get_existing_evidence(filtered, 100)

    if evidence.empty:
        st.info(
            "No existing evidence images are available for the selected "
            "filters. Deleted/missing images are ignored."
        )
    else:
        evidence_ids = evidence["id"].tolist()

        def evidence_label(event_id):
            row = evidence[evidence["id"] == event_id].iloc[0]

            if pd.isna(row["person_id"]):
                person = "N/A"
            else:
                person = str(int(row["person_id"]))

            return (
                f"Event #{event_id} | "
                f"{row['event_type']} | "
                f"Person {person}"
            )

        selected_id = st.selectbox(
            "Select event",
            evidence_ids,
            format_func=evidence_label,
        )

        selected = evidence[
            evidence["id"] == selected_id
        ].iloc[0]

        image_path = selected["resolved_path"]

        if image_path and image_path.exists():
            left, right = st.columns([2.2, 1], gap="large")

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
        'Administrative visibility into threats, incidents, evidence '
        'and security alert resolution.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not security_table_exists():
        st.markdown(
            """
<div class="alert-banner alert-info">
    🛡️ Security Center is ready, but the
    <b>security_alerts</b> table has not been created yet.
    Start the updated <b>main.py</b> once to initialize security logging.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="security-card">
    <b>Security pipeline status</b><br><br>
    🟢 Existing event database — available<br>
    🟡 Security alerts table — waiting for initialization<br>
    🟡 Threat notifications — waiting for configuration
</div>
""",
            unsafe_allow_html=True,
        )

    else:
        security_df = load_security_alerts()
        summary = security_summary(security_df)

        s1, s2, s3, s4, s5 = st.columns(5)

        s1.metric("Total Alerts", summary["total"])
        s2.metric("Open", summary["open"])
        s3.metric("Critical", summary["critical"])
        s4.metric("High", summary["high"])
        s5.metric("Resolved", summary["resolved"])

        if summary["critical"] > 0:
            st.markdown(
                f"""
<div class="alert-banner alert-danger">
    🚨 <b>CRITICAL SECURITY ATTENTION REQUIRED</b><br>
    {summary["critical"]} critical security alert(s) have been recorded.
</div>
""",
                unsafe_allow_html=True,
            )
        elif summary["open"] > 0:
            st.markdown(
                f"""
<div class="alert-banner alert-warning">
    ⚠️ {summary["open"]} security alert(s) require administrator review.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="alert-banner alert-success">
    ✓ Security system is clear. No open alerts.
</div>
""",
                unsafe_allow_html=True,
            )

        alerts_tab, security_evidence_tab, admin_tab = st.tabs(
            [
                "🚨 Alerts",
                "📸 Evidence",
                "🛡️ Administration",
            ]
        )

        # ----------------------------------------------------
        # ALERTS
        # ----------------------------------------------------

        with alerts_tab:
            if security_df.empty:
                st.info("No security alerts have been recorded yet.")
            else:
                c1, c2 = st.columns(2)

                with c1:
                    status_filter = st.selectbox(
                        "Status",
                        [
                            "ALL",
                            "OPEN",
                            "ACKNOWLEDGED",
                            "RESOLVED",
                        ],
                        key="security_status_filter",
                    )

                with c2:
                    severity_values = ["ALL"]

                    if "severity" in security_df.columns:
                        severity_values += sorted(
                            security_df["severity"]
                            .dropna()
                            .astype(str)
                            .str.upper()
                            .unique()
                            .tolist()
                        )

                    severity_filter = st.selectbox(
                        "Severity",
                        severity_values,
                        key="security_severity_filter",
                    )

                view_df = security_df.copy()

                if status_filter != "ALL" and "status" in view_df.columns:
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
                        col for col in preferred
                        if col in view_df.columns
                    ]

                    display_security = view_df[cols].copy()

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

                    left, right = st.columns([1, 1], gap="large")

                    with left:
                        st.markdown("#### Incident Details")

                        severity = selected_alert.get(
                            "severity", "UNKNOWN"
                        )
                        threat = selected_alert.get(
                            "threat_type", "UNKNOWN"
                        )
                        status = selected_alert.get(
                            "status", "UNKNOWN"
                        )
                        person = selected_alert.get(
                            "person_id", "N/A"
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
    <div class="security-number">
        #{selected_alert_id}
    </div>
    <div class="security-label">
        Security Incident
    </div>
    <br>
    <b>Threat:</b> {threat}<br>
    <b>Severity:</b> {severity}<br>
    <b>Person ID:</b> {person}<br>
    <b>Status:</b> {status}<br>
    <b>Created:</b> {created}
</div>
""",
                            unsafe_allow_html=True,
                        )

                        details = selected_alert.get(
                            "details", ""
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
                            "evidence_path", ""
                        )

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
                                "No evidence image is available for "
                                "this security alert."
                            )

                            if evidence_path:
                                st.caption(
                                    f"Recorded path: {evidence_path}"
                                )

                    st.markdown("#### Administrator Actions")

                    a1, a2 = st.columns(2)

                    current_status = str(
                        selected_alert.get(
                            "status", ""
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
                                "✓ Already Acknowledged",
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
                st.info("No security evidence available yet.")
            elif "evidence_path" not in security_df.columns:
                st.warning(
                    "The security database does not contain an "
                    "evidence_path column."
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
                    ]

                    if evidence_df.empty:
                        st.warning(
                            "The database references security evidence "
                            "files that are no longer available."
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
                            evidence_df["id"] == selected_evidence_id
                        ].iloc[0]

                        left, right = st.columns(
                            [2.2, 1],
                            gap="large",
                        )

                        with left:
                            st.image(
                                str(
                                    evidence_row["resolved_path"]
                                ),
                                use_container_width=True,
                                caption="Security evidence",
                            )

                        with right:
                            st.markdown("#### Incident")

                            st.write(
                                f"**Alert:** #{evidence_row['id']}"
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
        # ADMINISTRATION
        # ----------------------------------------------------

        with admin_tab:
            st.markdown("#### Administrator Controls")

            st.write(
                "Centralized administrative visibility over security "
                "incidents, evidence, notifications and database status."
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
                    st.success("🟢 Telegram configured")
                else:
                    st.warning("⚪ Telegram not configured")

            with n2:
                if email_configured:
                    st.success("🟢 Email configured")
                else:
                    st.warning("⚪ Email not configured")

            st.caption(
                "Keep credentials in Streamlit Secrets or environment "
                "variables. Never commit tokens or passwords to GitHub."
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

            st.markdown("##### Database Status")

            if DB_PATH.exists():
                st.success("🟢 SQLite database available")
                st.caption(str(DB_PATH))
            else:
                st.error("🔴 SQLite database unavailable")

            if security_table_exists():
                st.success("🟢 Security alerts table available")
            else:
                st.warning(
                    "🟡 Security alerts table not initialized"
                )

            st.markdown(
                """
<div class="security-card">
    <b>Administrator responsibility</b><br><br>
    The Security Center provides centralized visibility into
    security incidents, evidence, notification status and
    resolution state.<br><br>
    The original event database remains intact. Security alerts
    are handled separately so administration does not interfere
    with normal camera monitoring analytics.
</div>
""",
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    AI Camera Tracker • Analytics Command Center •
    SQLite event analytics • Security Monitoring
</div>
""",
    unsafe_allow_html=True,
)
