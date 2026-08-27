from pathlib import Path

from ultralytics import YOLO


class ObjectDetector:
    """
    General-purpose object detector.

    This is separate from the existing MediaPipe person/pose
    detector so existing person-specific functionality remains
    unchanged.
    """

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.50,
        iou=0.45,
    ):
        self.confidence = confidence
        self.iou = iou

        # Ultralytics automatically downloads the model the first
        # time it is used if it is not already available.
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Detect objects in a BGR OpenCV frame.

        Returns a list of dictionaries:

        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy)
        }
        """

        if frame is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        names = result.names

        for box in result.boxes:

            try:
                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                x1, y1, x2, y2 = (
                    coordinates
                )

            except (
                TypeError,
                ValueError,
                IndexError,
                AttributeError,
            ):
                continue

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )

            class_name = names.get(
                class_id,
                str(class_id),
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "center": (
                        cx,
                        cy,
                    ),
                }
            )

        return detections