from pathlib import Path
from datetime import datetime

import cv2

from database import log_event


ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "captured"


def capture_event(
    frame,
    person_id,
    event_type,
    activity=None,
    details="",
    category="events"
):

    folder = CAPTURE_DIR / category

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"{event_type.lower()}_"
        f"person_{person_id}_"
        f"{timestamp}.jpg"
    )

    image_path = folder / filename

    success = cv2.imwrite(
        str(image_path),
        frame
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {image_path}"
        )

    relative_path = str(
        image_path.relative_to(ROOT)
    )

    event_id = log_event(
        person_id=person_id,
        event_type=event_type,
        activity=activity,
        details=details,
        image_path=relative_path
    )

    print(
        f"[EVENT #{event_id}] "
        f"{event_type} | "
        f"Person {person_id} | "
        f"{relative_path}"
    )

    return event_id, relative_path