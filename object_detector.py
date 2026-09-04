"""
Compatibility object detector.

The main application now uses deepcamera_adapter.py as its single
project-local YOLO26 detection engine. This class remains available
for older modules that import ObjectDetector.
"""

from ultralytics import YOLO


class ObjectDetector:
    def __init__(
        self,
        model_path="yolo26n.pt",
        confidence=0.30,
        iou=0.45,
    ):
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.model = YOLO(model_path)

    def detect(self, frame):
        if frame is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        result = results[0]
        names = result.names

        for box in result.boxes:
            try:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )
            except Exception:
                continue

            x1, y1, x2, y2 = map(
                int,
                (x1, y1, x2, y2)
            )

            detections.append({
                "class_id": class_id,
                "class_name": names.get(
                    class_id,
                    str(class_id)
                ),
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "center": (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                ),
            })

        return detections
