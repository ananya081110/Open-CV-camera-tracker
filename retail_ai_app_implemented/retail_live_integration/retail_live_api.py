"""FastAPI server attached to the running Retail AI camera workers."""
from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from retail_live_state import LIVE_STATE

app = FastAPI(title="Retail AI Live API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health")
def health():
    s = LIVE_STATE.snapshot()
    return {"status": "ok", "ai_engine": "YOLO26 + local DeepCamera-style pipeline", "camera": s["camera_status"], "last_update": s["last_update"], "fleet": s["fleet_metrics"]}


@app.get("/api/v1/state")
def state():
    return LIVE_STATE.snapshot()


@app.get("/api/v1/overview")
def overview():
    s = LIVE_STATE.snapshot()
    return {"status": "live", "camera_id": s["camera_id"], "fps": s["fps"], **s["metrics"], "fleet_metrics": s["fleet_metrics"]}


@app.get("/api/v1/cameras")
def cameras():
    return LIVE_STATE.snapshot()["cameras"]


@app.get("/api/v1/fleet")
def fleet():
    s = LIVE_STATE.snapshot()
    return {"fleet_metrics": s["fleet_metrics"], "cameras": s["cameras"]}


@app.get("/api/v1/customers")
def customers():
    return LIVE_STATE.snapshot()["customers"]


@app.get("/api/v1/alerts")
def alerts():
    return LIVE_STATE.snapshot()["alerts"]


@app.get("/api/v1/events")
def events():
    return LIVE_STATE.snapshot()["events"]


@app.post("/api/v1/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: str):
    if not LIVE_STATE.acknowledge(alert_id):
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}


@app.get("/api/v1/analytics")
def analytics():
    s = LIVE_STATE.snapshot()
    return {"operational_metrics": s.get("operational_metrics", {}), "anomalies": s.get("anomalies", []), "zone_history": s.get("zone_history", []), "fleet_metrics": s.get("fleet_metrics", {})}


@app.get("/api/v1/notifications")
def notifications():
    return LIVE_STATE.snapshot().get("notification_status", {})


@app.get("/api/v1/zones")
def zones():
    s = LIVE_STATE.snapshot()
    return {"zones": s["zone_stats"], "staff_coverage": s["staff_coverage"]}


@app.get("/api/v1/cameras/{camera_id}")
def camera(camera_id: str):
    camera_state = LIVE_STATE.camera_snapshot(camera_id)
    if not camera_state:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera_state


@app.get("/api/v1/cameras/{camera_id}/video.mjpg")
def camera_video(camera_id: str):
    if not LIVE_STATE.camera_snapshot(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")

    def gen():
        while True:
            frame = LIVE_STATE.camera_frame(camera_id)
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.08)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/v1/video.mjpg")
def video():
    return camera_video(LIVE_STATE.camera_id)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(LIVE_STATE.snapshot())
            await asyncio.sleep(0.5)
    except Exception:
        pass


def start_live_api(host="127.0.0.1", port=8000):
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
