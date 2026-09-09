"""
Store-wide retail intelligence helpers.

The current POC can run one camera while exposing camera/zone
configuration and staff-aware decision interfaces that can be
used by a future multi-camera runner.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RetailCameraConfig:
    camera_id: str
    source: str
    zones: Dict[str, Tuple[float, float, float, float]]
    enabled: bool = True


class RetailCameraRegistry:
    """Configuration registry for multiple retail CCTV cameras."""

    def __init__(self):
        self.cameras: Dict[str, RetailCameraConfig] = {}

    def add_camera(
        self,
        camera_id,
        source,
        zones,
        enabled=True,
    ):
        self.cameras[str(camera_id)] = RetailCameraConfig(
            camera_id=str(camera_id),
            source=str(source),
            zones={
                str(name): tuple(float(v) for v in box)
                for name, box in zones.items()
                if isinstance(box, (list, tuple))
                and len(box) == 4
            },
            enabled=bool(enabled),
        )

    def get(self, camera_id):
        return self.cameras.get(str(camera_id))

    def active(self):
        return [
            camera
            for camera in self.cameras.values()
            if camera.enabled
        ]

    def status(self):
        return (
            f"Retail camera registry: "
            f"{len(self.active())}/{len(self.cameras)} cameras enabled"
        )
