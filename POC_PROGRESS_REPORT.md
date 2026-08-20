# AI Camera Tracker — POC Progress Report

## Implemented
1. OpenCV webcam capture.
2. MediaPipe Pose Landmarker.
3. Multi-person landmark extraction.
4. Simple centroid-based person IDs.
5. Standing/sitting/walking/falling rule classification.
6. Fall confirmation window to reduce one-frame false alarms.
7. Sitting, standing and zone dwell timers.
8. CSV event logging.
9. Fall snapshot capture.

## Why this is the right first version
The supplied research recommends OpenCV + MediaPipe Pose for the initial proof of concept, with a simple shoulder/hip-based fall rule and a basic tracker before moving to ByteTrack/DeepSORT and external notification APIs.

## Next iteration
- Test on webcam and sample videos.
- Tune thresholds and measure false positives.
- Add ByteTrack/DeepSORT.
- Add a temporal fall classifier.
- Add a dashboard.
- Add webhook/SMS notification.
