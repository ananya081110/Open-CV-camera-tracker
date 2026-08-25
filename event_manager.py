from pathlib import Path
from datetime import datetime

import cv2

from database import log_event


ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "captured"


def _timestamp():
    """
    Return a readable timestamp for console logging.
    """
    return datetime.now().isoformat(timespec="seconds")


def capture_event(
    frame,
    person_id,
    event_type,
    activity=None,
    details="",
    category="events"
):
    """
    Capture an event frame, save it to disk,
    log it in SQLite, and print a detailed event message.
    """

    # -------------------------------------------------
    # Timestamp
    # -------------------------------------------------

    now = datetime.now()

    console_timestamp = now.isoformat(
        timespec="seconds"
    )

    file_timestamp = now.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # -------------------------------------------------
    # Capture directory
    # -------------------------------------------------

    folder = CAPTURE_DIR / category

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Image filename
    # -------------------------------------------------

    filename = (
        f"{event_type.lower()}_"
        f"person_{person_id}_"
        f"{file_timestamp}.jpg"
    )

    image_path = folder / filename

    # -------------------------------------------------
    # Save image
    # -------------------------------------------------

    success = cv2.imwrite(
        str(image_path),
        frame
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {image_path}"
        )

    # -------------------------------------------------
    # Relative path for database
    # -------------------------------------------------

    relative_path = str(
        image_path.relative_to(ROOT)
    )

    # -------------------------------------------------
    # ALERT console message
    # -------------------------------------------------

    detail_text = details.strip()

    if detail_text:
        print(
            f"[ALERT] {console_timestamp} | "
            f"Person {person_id} | "
            f"{event_type} | "
            f"{detail_text}"
        )
    else:
        print(
            f"[ALERT] {console_timestamp} | "
            f"Person {person_id} | "
            f"{event_type}"
        )

    # -------------------------------------------------
    # SQLite event logging
    # -------------------------------------------------

    event_id = log_event(
        person_id=person_id,
        event_type=event_type,
        activity=activity,
        details=details,
        image_path=relative_path
    )

    # -------------------------------------------------
    # EVENT message
    # -------------------------------------------------

    print(
        f"[EVENT #{event_id}] "
        f"{event_type} | "
        f"Person {person_id} | "
        f"{relative_path}"
    )

    return event_id, relative_path


def log_alert(
    person_id,
    event_type,
    details="",
    activity=None
):
    """
    Log an event without capturing an image.

    Useful for events where image capture is not required.
    """

    timestamp = _timestamp()

    # -------------------------------------------------
    # Console alert
    # -------------------------------------------------

    if details:
        print(
            f"[ALERT] {timestamp} | "
            f"Person {person_id} | "
            f"{event_type} | "
            f"{details}"
        )
    else:
        print(
            f"[ALERT] {timestamp} | "
            f"Person {person_id} | "
            f"{event_type}"
        )

    # -------------------------------------------------
    # Database logging
    # -------------------------------------------------

    event_id = log_event(
        person_id=person_id,
        event_type=event_type,
        activity=activity,
        details=details,
        image_path=None
    )

    print(
        f"[EVENT #{event_id}] "
        f"{event_type} | "
        f"Person {person_id} | "
        f"database-only event"
    )

    return event_id