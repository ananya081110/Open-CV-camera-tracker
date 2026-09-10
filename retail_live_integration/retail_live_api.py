"""FastAPI server attached to the running AI camera process."""
from __future__ import annotations
import asyncio, time
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from retail_live_state import LIVE_STATE

app = FastAPI(title="Retail AI Live API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/v1/health")
def health():
    s = LIVE_STATE.snapshot()
    return {"status": "ok", "ai_engine": "YOLO26 + local DeepCamera-style pipeline", "camera": s["camera_status"], "last_update": s["last_update"]}

@app.get("/api/v1/overview")
def overview():
    s = LIVE_STATE.snapshot()
    return {"status": "live", "camera_id": s["camera_id"], "fps": s["fps"], **s["metrics"]}

@app.get("/api/v1/cameras")
def cameras():
    s = LIVE_STATE.snapshot()
    return [{"camera_id": s["camera_id"], "status": s["camera_status"], "fps": s["fps"]}]

@app.get("/api/v1/customers")
def customers():
    return LIVE_STATE.snapshot()["customers"]

@app.get("/api/v1/alerts")
def alerts():
    return LIVE_STATE.snapshot()["alerts"]

@app.get("/api/v1/events")
def events():
    return LIVE_STATE.snapshot()["events"]

@app.get("/api/v1/cameras/{camera_id}")
def camera(camera_id: str):
    s = LIVE_STATE.snapshot()
    return {"camera_id": camera_id, "status": s["camera_status"], "fps": s["fps"], "customers": s["customers"]}

@app.get("/api/v1/video.mjpg")
def video():
    def gen():
        while True:
            with LIVE_STATE.lock:
                frame = LIVE_STATE.frame_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

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
