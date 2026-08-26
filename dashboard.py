import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "logs" / "camera_events.db"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Camera Tracker",
    page_icon="📹",
    layout="wide",
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
            ORDER BY timestamp ASC
            """,
            conn,
        )

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

    return df


def get_latest_event_image():
    if not DB_PATH.exists():
        return None

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT image_path, timestamp, event_type, person_id, activity
            FROM events
            WHERE image_path IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        return None

    image_path = ROOT / row[0]

    if not image_path.exists():
        return None

    return {
        "path": image_path,
        "timestamp": row[1],
        "event_type": row[2],
        "person_id": row[3],
        "activity": row[4],
    }


# ============================================================
# ANALYTICS HELPERS
# ============================================================

def posture_durations(events):
    """
    Estimate posture duration from POSTURE_CHANGE events.

    The duration is attributed to the posture that was active
    between one posture-change event and the next.

    This works with the existing event schema and does not
    require changing database.py.
    """
    if events.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "activity",
                "start",
                "end",
                "duration_seconds",
            ]
        )

    posture_events = events[
        events["event_type"].astype(str).str.upper().eq("POSTURE_CHANGE")
    ].copy()

    if posture_events.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "activity",
                "start",
                "end",
                "duration_seconds",
            ]
        )

    posture_events = posture_events.sort_values(
        ["person_id", "timestamp"]
    )

    rows = []

    for person_id, group in posture_events.groupby(
        "person_id",
        dropna=False,
    ):
        group = group.reset_index(drop=True)

        for i in range(len(group) - 1):
            current = group.iloc[i]
            following = group.iloc[i + 1]

            if pd.isna(current["timestamp"]) or pd.isna(
                following["timestamp"]
            ):
                continue

            duration = (
                following["timestamp"] - current["timestamp"]
            ).total_seconds()

            # Ignore obviously corrupted/huge intervals.
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

    return pd.DataFrame(rows)


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


def event_count(events, name):
    return int(
        (
            events["event_type"]
            .astype(str)
            .str.upper()
            == name.upper()
        ).sum()
    )


# ============================================================
# LOAD
# ============================================================

df = load_events()

if df.empty:
    st.title("📹 AI Camera Tracker")
    st.warning(
        "No events found yet. Start main.py and generate monitoring events."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📹 AI Camera Tracker Dashboard")
st.caption(
    "Monitoring, posture analytics, event history and captured evidence"
)

st.sidebar.header("Dashboard")

if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# DATE FILTER
# ============================================================

valid_dates = df["timestamp"].dropna()

if not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates

        filtered = df[
            (
                df["timestamp"].dt.date >= start_date
            )
            & (
                df["timestamp"].dt.date <= end_date
            )
        ].copy()
    else:
        filtered = df.copy()
else:
    filtered = df.copy()


# ============================================================
# EVENT FILTER
# ============================================================

event_types = sorted(
    filtered["event_type"]
    .dropna()
    .astype(str)
    .unique()
)

selected_event_types = st.sidebar.multiselect(
    "Event types",
    event_types,
    default=list(event_types),
)

filtered = filtered[
    filtered["event_type"].astype(str).isin(
        selected_event_types
    )
]


# ============================================================
# TABS
# ============================================================

overview_tab, live_tab, posture_tab, reports_tab, events_tab = st.tabs(
    [
        "📊 Overview",
        "📡 Live Monitor",
        "🧍 Posture Analytics",
        "📅 Reports",
        "📝 Events",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    total_events = len(filtered)

    people = (
        filtered["person_id"]
        .dropna()
        .nunique()
    )

    entries = event_count(
        filtered,
        "ZONE_ENTRY",
    )

    exits = event_count(
        filtered,
        "ZONE_EXIT",
    )

    limited = event_count(
        filtered,
        "LIMITED_VIEW",
    )

    falls = int(
        filtered["event_type"]
        .astype(str)
        .str.contains(
            "FALL",
            case=False,
            na=False,
        )
        .sum()
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Total Events", total_events)
    c2.metric("People", people)
    c3.metric("Zone Entries", entries)
    c4.metric("Zone Exits", exits)
    c5.metric("Limited View", limited)
    c6.metric("Falls", falls)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Event Distribution")

        counts = (
            filtered["event_type"]
            .value_counts()
            .rename_axis("Event")
            .reset_index(name="Count")
        )

        if not counts.empty:
            st.bar_chart(
                counts.set_index("Event")
            )

    with right:
        st.subheader("Activity Distribution")

        activity_counts = (
            filtered["activity"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .value_counts()
            .rename_axis("Activity")
            .reset_index(name="Count")
        )

        if not activity_counts.empty:
            st.bar_chart(
                activity_counts.set_index("Activity")
            )

    st.divider()

    st.subheader("Event Timeline")

    timeline = filtered.dropna(
        subset=["timestamp"]
    ).copy()

    if not timeline.empty:
        timeline_counts = (
            timeline
            .set_index("timestamp")
            .resample("1min")
            .size()
        )

        st.line_chart(timeline_counts)

    st.divider()

    st.subheader("Latest Activity")

    latest = filtered.sort_values(
        "timestamp",
        ascending=False,
    ).head(10).copy()

    latest["timestamp"] = latest[
        "timestamp"
    ].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(
        latest[
            [
                "timestamp",
                "person_id",
                "event_type",
                "activity",
                "details",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# LIVE MONITOR
# ============================================================

with live_tab:

    st.subheader("Latest Monitoring Snapshot")

    st.info(
        "This view uses the latest automatically captured event image. "
        "It does not open a second camera stream, so it can run safely "
        "while main.py owns the webcam."
    )

    latest_image = get_latest_event_image()

    if latest_image:

        left, right = st.columns([2, 1])

        with left:
            st.image(
                str(latest_image["path"]),
                caption=(
                    f"{latest_image['event_type']} | "
                    f"Person {latest_image['person_id']}"
                ),
                use_container_width=True,
            )

        with right:
            st.subheader("Latest Event")

            st.write(
                f"**Event:** {latest_image['event_type']}"
            )

            st.write(
                f"**Person:** {latest_image['person_id']}"
            )

            st.write(
                f"**Activity:** "
                f"{latest_image['activity'] or 'N/A'}"
            )

            st.write(
                f"**Timestamp:** "
                f"{latest_image['timestamp']}"
            )

            st.write(
                f"**Image:** "
                f"`{latest_image['path'].relative_to(ROOT)}`"
            )

    else:
        st.warning(
            "No captured event image is available yet."
        )

    st.divider()

    st.subheader("Current Monitoring State")

    latest_per_person = (
        filtered
        .sort_values("timestamp")
        .dropna(subset=["person_id"])
        .groupby("person_id", as_index=False)
        .tail(1)
    )

    if latest_per_person.empty:
        st.info("No active person state available.")
    else:
        state_display = latest_per_person[
            [
                "person_id",
                "timestamp",
                "event_type",
                "activity",
                "details",
            ]
        ].copy()

        state_display["timestamp"] = state_display[
            "timestamp"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            state_display.sort_values(
                "person_id"
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# POSTURE ANALYTICS
# ============================================================

with posture_tab:

    st.subheader("Posture Duration Analysis")

    durations = posture_durations(df)

    if durations.empty:

        st.info(
            "No POSTURE_CHANGE events are available yet. "
            "Once posture changes are logged, duration analytics "
            "will appear here."
        )

    else:

        # Apply selected date range where possible.
        posture_filtered = durations.copy()

        if "selected_dates" in locals():
            if (
                isinstance(selected_dates, tuple)
                and len(selected_dates) == 2
            ):
                posture_filtered = posture_filtered[
                    (
                        posture_filtered["start"].dt.date
                        >= selected_dates[0]
                    )
                    & (
                        posture_filtered["start"].dt.date
                        <= selected_dates[1]
                    )
                ]

        total_by_activity = (
            posture_filtered
            .groupby("activity")["duration_seconds"]
            .sum()
            .sort_values(ascending=False)
        )

        c1, c2, c3 = st.columns(3)

        sitting_seconds = float(
            total_by_activity.get(
                "SITTING",
                0,
            )
        )

        standing_seconds = float(
            total_by_activity.get(
                "STANDING",
                0,
            )
        )

        unknown_seconds = float(
            total_by_activity.get(
                "UNKNOWN",
                0,
            )
        )

        c1.metric(
            "Sitting",
            format_duration(sitting_seconds),
        )

        c2.metric(
            "Standing",
            format_duration(standing_seconds),
        )

        c3.metric(
            "Unknown",
            format_duration(unknown_seconds),
        )

        st.divider()

        chart_data = (
            total_by_activity
            .rename("seconds")
            .to_frame()
        )

        st.subheader("Time by Posture")

        if not chart_data.empty:
            st.bar_chart(chart_data)

        st.divider()

        st.subheader("Posture Intervals")

        posture_table = posture_filtered.copy()

        posture_table["Duration"] = posture_table[
            "duration_seconds"
        ].apply(format_duration)

        posture_table["Start"] = posture_table[
            "start"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")

        posture_table["End"] = posture_table[
            "end"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            posture_table[
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
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DAILY / WEEKLY REPORTS
# ============================================================

with reports_tab:

    st.subheader("Monitoring Reports")

    report_df = filtered.copy()

    report_df["date"] = (
        report_df["timestamp"]
        .dt.date
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

    if not daily.empty:

        daily["zone_entries"] = (
            report_df[
                report_df["event_type"]
                .astype(str)
                .str.upper()
                .eq("ZONE_ENTRY")
            ]
            .groupby("date")
            .size()
            .reindex(
                daily["date"],
                fill_value=0,
            )
            .values
        )

        daily["zone_exits"] = (
            report_df[
                report_df["event_type"]
                .astype(str)
                .str.upper()
                .eq("ZONE_EXIT")
            ]
            .groupby("date")
            .size()
            .reindex(
                daily["date"],
                fill_value=0,
            )
            .values
        )

        daily["limited_view"] = (
            report_df[
                report_df["event_type"]
                .astype(str)
                .str.upper()
                .eq("LIMITED_VIEW")
            ]
            .groupby("date")
            .size()
            .reindex(
                daily["date"],
                fill_value=0,
            )
            .values
        )

        daily["falls"] = (
            report_df[
                report_df["event_type"]
                .astype(str)
                .str.contains(
                    "FALL",
                    case=False,
                    na=False,
                )
            ]
            .groupby("date")
            .size()
            .reindex(
                daily["date"],
                fill_value=0,
            )
            .values
        )

        st.subheader("Daily Summary")

        st.dataframe(
            daily.sort_values(
                "date",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        st.subheader("Weekly Event Trend")

        weekly = (
            report_df
            .set_index("timestamp")
            .resample("W")
            .size()
        )

        if not weekly.empty:
            st.line_chart(weekly)

        st.divider()

        st.subheader("Report Summary")

        total_days = daily["date"].nunique()

        average_events = (
            daily["total_events"].mean()
            if not daily.empty
            else 0
        )

        peak_day = (
            daily.loc[
                daily["total_events"].idxmax(),
                "date",
            ]
            if not daily.empty
            else "N/A"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Monitoring Days",
            int(total_days),
        )

        c2.metric(
            "Avg Events / Day",
            f"{average_events:.1f}",
        )

        c3.metric(
            "Most Active Day",
            str(peak_day),
        )

    else:
        st.info("No report data available.")


# ============================================================
# EVENTS
# ============================================================

with events_tab:

    st.subheader("Complete Event Log")

    display = filtered.copy()

    display["timestamp"] = display[
        "timestamp"
    ].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(
        display[
            [
                "id",
                "timestamp",
                "person_id",
                "event_type",
                "activity",
                "details",
                "image_path",
            ]
        ].sort_values(
            "id",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Captured Evidence")

    evidence = filtered[
        filtered["image_path"].notna()
    ].copy()

    if evidence.empty:
        st.info("No captured images for the selected filters.")
    else:

        selected_id = st.selectbox(
            "Select event",
            evidence["id"].tolist(),
            format_func=lambda event_id: (
                f"Event #{event_id} | "
                f"{evidence.loc[evidence['id'] == event_id, 'event_type'].iloc[0]} | "
                f"Person "
                f"{evidence.loc[evidence['id'] == event_id, 'person_id'].iloc[0]}"
            ),
        )

        selected = evidence[
            evidence["id"] == selected_id
        ].iloc[0]

        image_path = ROOT / selected["image_path"]

        if image_path.exists():

            left, right = st.columns([2, 1])

            with left:
                st.image(
                    str(image_path),
                    use_container_width=True,
                )

            with right:
                st.write(
                    f"**Timestamp:** "
                    f"{selected['timestamp']}"
                )
                st.write(
                    f"**Event:** "
                    f"{selected['event_type']}"
                )
                st.write(
                    f"**Person:** "
                    f"{selected['person_id']}"
                )
                st.write(
                    f"**Activity:** "
                    f"{selected['activity'] or 'N/A'}"
                )
                st.write(
                    f"**Details:** "
                    f"{selected['details'] or 'N/A'}"
                )

        else:
            st.warning(
                "The database references an image that no longer exists."
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption(
    f"Database: {DB_PATH}"
)
st.sidebar.caption(
    "AI Camera Tracker • Analytics Dashboard"
)