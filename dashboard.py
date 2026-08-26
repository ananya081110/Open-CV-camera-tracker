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
# MODERN MNC-STYLE UI
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

        section[data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #1f2937;
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        section[data-testid="stSidebar"] .stButton button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #374151;
            background: #1f2937;
            color: white;
        }

        section[data-testid="stSidebar"] .stButton button:hover {
            border-color: #60a5fa;
            background: #263449;
        }

        .app-header {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .app-title {
            font-size: 30px;
            font-weight: 750;
            color: #111827;
            letter-spacing: -0.025em;
        }

        .app-subtitle {
            margin-top: 4px;
            color: #6b7280;
            font-size: 14px;
        }

        .section-title {
            font-size: 19px;
            font-weight: 700;
            color: #111827;
            margin-top: 8px;
            margin-bottom: 3px;
        }

        .section-subtitle {
            color: #6b7280;
            font-size: 13px;
            margin-bottom: 14px;
        }

        .status-online {
            display: inline-block;
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 12px;
            font-weight: 650;
        }

        .status-offline {
            display: inline-block;
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
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
            color: #991b1b;
        }

        .alert-warning {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }

        .alert-success {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }

        .info-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 7px rgba(15, 23, 42, 0.035);
        }

        .footer {
            color: #9ca3af;
            font-size: 11px;
            text-align: center;
            padding-top: 28px;
        }

        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }

        button[data-baseweb="tab"] {
            font-weight: 650;
            font-size: 13px;
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
    """Load and normalize the existing SQLite event database."""
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


# ============================================================
# IMAGE HELPERS
# ============================================================

def resolve_image_path(relative_path):
    """
    Resolve an image path from the database.

    Old database records can reference deleted files, so a
    missing image is treated as unavailable rather than fatal.
    """
    if not relative_path:
        return None

    try:
        path = ROOT / str(relative_path)
        if path.exists() and path.is_file():
            return path
    except (OSError, TypeError, ValueError):
        pass

    return None


def get_existing_evidence(events, limit=100):
    """Return recent evidence rows whose image files still exist."""
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

    evidence["resolved_path"] = evidence[
        "image_path"
    ].apply(resolve_image_path)

    return evidence[
        evidence["resolved_path"].notna()
    ].copy()


def get_latest_event_image():
    """Find the newest database event that points to a real image."""
    evidence = get_existing_evidence(load_events(), limit=100)

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

    event_series = (
        events["event_type"]
        .astype(str)
        .str.upper()
    )

    return int(
        event_series.str.contains(
            r"FALL|ALERT",
            regex=True,
            na=False,
        ).sum()
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
    """Latest known event/activity for every tracked person."""
    if events.empty:
        return pd.DataFrame()

    usable = events.dropna(subset=["person_id"]).copy()

    if usable.empty:
        return pd.DataFrame()

    latest = (
        usable
        .sort_values(["timestamp", "id"])
        .groupby("person_id", as_index=False)
        .tail(1)
    )

    return latest.sort_values("person_id")


def posture_durations(events):
    """
    Estimate time spent in each posture from POSTURE_CHANGE events.

    A posture is considered active from one posture-change event
    until the next posture-change event for the same person.
    """
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

            if duration <= 0 or duration > 24 * 60 * 60:
                continue

            activity = str(
                current["activity"] or "UNKNOWN"
            ).upper()

            rows.append(
                {
                    "person_id": person_id,
                    "activity": activity,
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
# LOAD DATA
# ============================================================

df = load_events()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:20px;font-weight:750;">
            AI Camera Tracker
        </div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">
            Monitoring Command Center
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    if st.button(
        "🔄 Refresh Dashboard",
        use_container_width=True,
    ):
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

    filtered = filter_by_date(
        df,
        selected_dates,
    )

    event_types = sorted(
        filtered["event_type"]
        .dropna()
        .astype(str)
        .unique()
    )

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
        <div style="font-size:11px;color:#9ca3af;line-height:1.7;">
            <b>Data source</b><br>
            SQLite event database<br><br>
            <b>Pipeline</b><br>
            Pose Detection → Tracking → Classification →
            Event Logging → Analytics
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EMPTY DATABASE STATE
# ============================================================

if df.empty:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">📹 AI Camera Tracker</div>
            <div class="app-subtitle">
                Intelligent monitoring & analytics command center
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
# KPI SUMMARY
# ============================================================

total_events = len(filtered)

people = (
    filtered["person_id"]
    .dropna()
    .nunique()
)

entries = event_count(filtered, "ZONE_ENTRY")
exits = event_count(filtered, "ZONE_EXIT")
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
# ALERT SUMMARY
# ============================================================

if falls > 0:
    st.markdown(
        f"""
        <div class="alert-banner alert-danger">
            🚨 {falls} fall / alert event(s) detected in the selected
            monitoring period. Review Events & Evidence immediately.
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
            ✓ No fall or critical alert events detected in the
            selected monitoring period.
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
) = st.tabs(
    [
        "📊 Overview",
        "📡 Live Monitor",
        "🧍 Posture Analytics",
        "📅 Reports",
        "📝 Events & Evidence",
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
            st.bar_chart(
                counts,
                use_container_width=True,
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

        st.line_chart(
            timeline_counts,
            use_container_width=True,
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
            durations
            .groupby("activity")["duration_seconds"]
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

        chart_data = total_by_activity.rename(
            "seconds"
        ).to_frame()

        if chart_data.empty:
            st.info("No posture duration data.")
        else:
            st.bar_chart(
                chart_data,
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### Posture Intervals")

        posture_table = durations.copy()

        posture_table["Duration"] = posture_table[
            "duration_seconds"
        ].apply(format_duration)

        posture_table["Start"] = posture_table[
            "start"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")

        posture_table["End"] = posture_table[
            "end"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")

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
        report_df["date"] = report_df["timestamp"].dt.date

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
            mask = report_df["event_type"].str.contains(
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
        display["timestamp"] = (
            display["timestamp"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
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
        limit=100,
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

            person = (
                "N/A"
                if pd.isna(row["person_id"])
                else str(int(row["person_id"]))
            )

            return (
                f"Event #{event_id} | "
                f"{row['event_type']} | "
                f"Person {person}"
            )

        selected_id = st.selectbox(
            "Select event",
            options=evidence_ids,
            format_func=evidence_label,
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
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'AI Camera Tracker • Analytics Command Center • '
    'SQLite event analytics'
    '</div>',
    unsafe_allow_html=True,
)
