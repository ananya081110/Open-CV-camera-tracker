# Retail AI Desktop Application

The Retail AI project can now be launched as a single desktop application during development.

## Architecture

```text
Retail AI Desktop (Electron)
        |
        +--> React Command Center (Vite)
        |
        +--> Python AI / FastAPI backend
                |
                +--> Camera + Detection + Tracking
                +--> Zones + Customer Intelligence
                +--> Staff Coverage + Alerts
                +--> AI Store Manager + Assistant
```

## Start

```bash
cd ~/Downloads/ai_camera_tracker/retail_ai_app_implemented/retail_dashboard
npm install
npm run desktop
```

The desktop launcher starts both the UI and the Python backend and closes the child processes when the app exits.

## Notes

- The launcher expects the existing project structure and `.venv` at the project root.
- Keep secrets such as Twilio credentials in environment variables; never commit them.
- This is the desktop development shell. A distributable macOS `.app` should be the next packaging step after the local flow is validated.
