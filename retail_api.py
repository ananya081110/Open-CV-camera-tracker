"""FastAPI backend for the retail AI intelligence platform.

This layer is intentionally decoupled from the existing Streamlit dashboard.
It exposes store status, cameras, customers, alerts and live events to a
future React frontend without changing the existing AI/security pipeline.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


class RetailAPIServer:
    def __init__(self, engine=None, pipeline=None):
        self.engine = engine
        self.pipeline = pipeline
        self._clients: Set[WebSocket] = set()
        self._started_at = time.time()

    def camera_status(self) -> Dict[str, Any]:
        if self.engine is None:
            return {"cameras": {}, "global_reid": {"active_customers": 0}}
        return self.engine.status()

    def customer_summary(self) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            return []
        events = self.pipeline.recent_events(500)
        latest: Dict[str, Dict[str, Any]] = {}
        for event in events:
            if event.get("event") != "customer_observation":
                continue
            customer_id = event.get("customer_id")
            if customer_id:
                latest[str(customer_id)] = dict(event)
        return list(latest.values())

    def alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            return []
        return [
            e for e in self.pipeline.recent_events(max(limit * 4, 1000))
            if e.get("event") == "retail_alert"
        ][-limit:][::-1]

    def overview(self) -> Dict[str, Any]:
        camera_data = self.camera_status()
        cameras = camera_data.get("cameras", {})
        connected = sum(1 for c in cameras.values() if c.get("connected"))
        running = sum(1 for c in cameras.values() if c.get("running"))
        customers = self.customer_summary()
        alerts = self.alerts(100)
        high_intent = sum(1 for c in customers if c.get("intent_level") == "HIGH")
        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "cameras_total": len(cameras),
            "cameras_connected": connected,
            "cameras_running": running,
            "active_customers": len(customers),
            "high_intent_customers": high_intent,
            "active_alerts": len(alerts),
            "timestamp": time.time(),
        }

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        self._clients.difference_update(dead)


def create_app(engine=None, pipeline=None) -> FastAPI:
    server = RetailAPIServer(engine=engine, pipeline=pipeline)

    app = FastAPI(
        title="Retail AI Intelligence API",
        version="1.0.0",
        description="Real-time retail CCTV intelligence backend for the React dashboard.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "retail-ai-api", "timestamp": time.time()}

    @app.get("/api/v1/overview")
    async def overview():
        return server.overview()

    @app.get("/api/v1/cameras")
    async def cameras():
        return server.camera_status()

    @app.get("/api/v1/customers")
    async def customers():
        return {"customers": server.customer_summary()}

    @app.get("/api/v1/alerts")
    async def alerts(limit: int = 50):
        limit = max(1, min(limit, 500))
        return {"alerts": server.alerts(limit)}

    @app.get("/api/v1/events")
    async def events(limit: int = 100):
        if server.pipeline is None:
            return {"events": []}
        limit = max(1, min(limit, 1000))
        return {"events": server.pipeline.recent_events(limit)}

    @app.get("/api/v1/cameras/{camera_id}")
    async def camera(camera_id: str):
        data = server.camera_status().get("cameras", {})
        if camera_id not in data:
            raise HTTPException(status_code=404, detail="Camera not found")
        return {"camera_id": camera_id, **data[camera_id]}

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket):
        await websocket.accept()
        server._clients.add(websocket)
        try:
            while True:
                await websocket.send_json({
                    "type": "snapshot",
                    "overview": server.overview(),
                    "cameras": server.camera_status(),
                    "alerts": server.alerts(10),
                    "timestamp": time.time(),
                })
                await asyncio.sleep(2)
        except (WebSocketDisconnect, Exception):
            server._clients.discard(websocket)

    app.state.retail_server = server
    return app


# Development-safe standalone API. Production startup should inject the live
# multi-camera engine and event pipeline from the retail runtime.
app = create_app()


__all__ = ["RetailAPIServer", "create_app", "app"]

