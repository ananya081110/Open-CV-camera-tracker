# Retail AI Intelligence Dashboard

React/Vite frontend for the Retail AI FastAPI backend.

## Run

```bash
npm install
npm run dev
```

The frontend expects the API at `http://127.0.0.1:8000` by default.
Override it with:

```bash
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

The current UI consumes camera status, overview metrics and alerts. Actual CCTV video tiles will be connected after the streaming layer (WebRTC/HLS) is added.
