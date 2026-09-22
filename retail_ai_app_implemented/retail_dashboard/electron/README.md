# Retail AI Desktop App

This desktop shell launches the existing Python AI backend and React/Vite command center together in one Electron window.

## Run locally

From `retail_ai_app_implemented/retail_dashboard`:

```bash
npm install
npm run desktop
```

Replace the accidental leading `a` above with:

```bash
npm run desktop
```

The app starts:
- Python AI backend on `http://127.0.0.1:8000`
- React/Vite UI on `http://127.0.0.1:5173`
- Electron desktop window wrapping the command center

The launcher automatically uses `../../.venv/bin/python3` when that virtual environment exists.

## Production direction

`npm run build` creates the React production bundle. For a distributable `.app`, the next packaging step is to bundle the Python AI service with PyInstaller and then configure `electron-builder` to ship the Python executable and the React `dist/` folder. This repository intentionally keeps that packaging step separate so the development environment remains simple and the existing AI dependencies are not copied into Git.
