import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "logs" / "camera_events.db"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Camera Tracker Dashboard",
    page_icon="📹",
    layout="wide"
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


@st.cache_data(ttl=2)
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
            ORDER BY timestamp DESC
            """,
            conn
        )

    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    return df


# ============================================================
# HEADER
# ============================================================

st.title("📹 AI Camera Tracker")
st.caption(
    "Real-time monitoring, posture analysis and event analytics"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

df = load_events()


if df.empty:

    st.warning(
        "No events found. Start main.py and generate monitoring events."
    )

    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

event_types = sorted(
    df["event_type"]
    .dropna()
    .unique()
    .tolist()
)

selected_events = st.sidebar.multiselect(
    "Event Type",
    event_types,
    default=event_types
)

activities = sorted(
    df["activity"]
    .dropna()
    .unique()
    .tolist()
)

selected_activities = st.sidebar.multiselect(
    "Activity",
    activities,
    default=activities
)


# Apply filters

filtered_df = df[
    df["event_type"].isin(selected_events)
]

if selected_activities:
    filtered_df = filtered_df[
        filtered_df["activity"].isin(selected_activities)
    ]


# ============================================================
# KPI SECTION
# ============================================================

total_events = len(filtered_df)

unique_people = (
    filtered_df["person_id"]
    .dropna()
    .nunique()
)

zone_entries = len(
    filtered_df[
        filtered_df["event_type"] == "ZONE_ENTRY"
    ]
)

zone_exits = len(
    filtered_df[
        filtered_df["event_type"] == "ZONE_EXIT"
    ]
)

limited_views = len(
    filtered_df[
        filtered_df["event_type"] == "LIMITED_VIEW"
    ]
)

fall_events = len(
    filtered_df[
        filtered_df["event_type"].str.contains(
            "FALL",
            case=False,
            na=False
        )
    ]
)


col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Total Events",
        total_events
    )

with col2:
    st.metric(
        "People Detected",
        unique_people
    )

with col3:
    st.metric(
        "Zone Entries",
        zone_entries
    )

with col4:
    st.metric(
        "Zone Exits",
        zone_exits
    )

with col5:
    st.metric(
        "Limited View",
        limited_views
    )

with col6:
    st.metric(
        "Falls",
        fall_events
    )


st.divider()


# ============================================================
# EVENT ANALYTICS
# ============================================================

left, right = st.columns(2)


# -------------------------
# Event distribution
# -------------------------

with left:

    st.subheader("Event Distribution")

    event_counts = (
        filtered_df["event_type"]
        .value_counts()
        .rename_axis("Event")
        .reset_index(name="Count")
    )

    if not event_counts.empty:

        st.bar_chart(
            event_counts.set_index("Event")
        )


# -------------------------
# Activity distribution
# -------------------------

with right:

    st.subheader("Activity Distribution")

    activity_counts = (
        filtered_df["activity"]
        .fillna("UNKNOWN")
        .value_counts()
        .rename_axis("Activity")
        .reset_index(name="Count")
    )

    if not activity_counts.empty:

        st.bar_chart(
            activity_counts.set_index("Activity")
        )


st.divider()


# ============================================================
# EVENT TIMELINE
# ============================================================

st.subheader("Event Timeline")

timeline = filtered_df.copy()

timeline["time"] = timeline["timestamp"]

timeline_counts = (
    timeline
    .set_index("time")
    .resample("1min")
    .size()
)

if not timeline_counts.empty:

    st.line_chart(
        timeline_counts
    )


st.divider()


# ============================================================
# RECENT EVENTS
# ============================================================

st.subheader("Recent Events")

display_df = filtered_df[
    [
        "id",
        "timestamp",
        "person_id",
        "event_type",
        "activity",
        "details"
    ]
].copy()

display_df["timestamp"] = display_df[
    "timestamp"
].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

st.dataframe(
    display_df.head(50),
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# CAPTURED EVENT IMAGE
# ============================================================

st.subheader("Captured Event")

if not filtered_df.empty:

    image_options = filtered_df[
        filtered_df["image_path"].notna()
    ].copy()

    if not image_options.empty:

        selected_index = st.selectbox(
            "Select an event",
            image_options.index,
            format_func=lambda i:
                (
                    f"#{image_options.loc[i, 'id']} | "
                    f"{image_options.loc[i, 'event_type']} | "
                    f"Person "
                    f"{image_options.loc[i, 'person_id']} | "
                    f"{image_options.loc[i, 'timestamp']}"
                )
        )

        selected_event = image_options.loc[
            selected_index
        ]

        image_path = (
            ROOT /
            selected_event["image_path"]
        )

        col1, col2 = st.columns([2, 1])

        with col1:

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption=(
                        f"{selected_event['event_type']} "
                        f"| Person "
                        f"{selected_event['person_id']}"
                    ),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Captured image could not be found."
                )

        with col2:

            st.write("### Event Details")

            st.write(
                f"**Event:** "
                f"{selected_event['event_type']}"
            )

            st.write(
                f"**Person:** "
                f"{selected_event['person_id']}"
            )

            st.write(
                f"**Activity:** "
                f"{selected_event['activity'] or 'N/A'}"
            )

            st.write(
                f"**Time:** "
                f"{selected_event['timestamp']}"
            )

            st.write(
                f"**Details:** "
                f"{selected_event['details'] or 'N/A'}"
            )

            st.write(
                f"**Image:** "
                f"`{selected_event['image_path']}`"
            )


# ============================================================
# AUTO REFRESH
# ============================================================

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Dashboard"):

    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Dashboard reads directly from SQLite."
)