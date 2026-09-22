const { app, BrowserWindow, dialog, shell } = require('electron');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');

const DASHBOARD_DIR = path.resolve(__dirname, '..');
const PROJECT_ROOT = path.resolve(DASHBOARD_DIR, '..', '..');
const BACKEND_SCRIPT = path.join(PROJECT_ROOT, 'retail_ai_app_implemented', 'retail_live_integration', 'main_retail_store_intelligence.py');
const VENV_PYTHON = path.join(PROJECT_ROOT, '.venv', 'bin', 'python3');
const PYTHON = fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python3';
const VITE_URL = 'http://127.0.0.1:5173';
const API_URL = 'http://127.0.0.1:8000';

let mainWindow;
let backendProcess;
let viteProcess;

function waitForUrl(url, timeoutMs = 30000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      const req = http.get(url, res => {
        res.resume();
        if (res.statusCode && res.statusCode < 500) return resolve(true);
        retry();
      });
      req.on('error', retry);
      req.setTimeout(1200, () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() - started > timeoutMs) return reject(new Error(`Timed out waiting for ${url}`));
      setTimeout(check, 350);
    };
    check();
  });
}

function startBackend() {
  if (!fs.existsSync(BACKEND_SCRIPT)) {
    throw new Error(`Backend script not found:\n${BACKEND_SCRIPT}`);
  }
  backendProcess = spawn(PYTHON, [BACKEND_SCRIPT], {
    cwd: PROJECT_ROOT,
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  backendProcess.stdout.on('data', d => console.log(`[AI] ${d}`));
  backendProcess.stderr.on('data', d => console.error(`[AI] ${d}`));
  backendProcess.on('exit', code => console.log(`[AI] backend exited (${code})`));
}

function startVite() {
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  viteProcess = spawn(npm, ['run', 'dev', '--', '--host', '127.0.0.1'], {
    cwd: DASHBOARD_DIR,
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  viteProcess.stdout.on('data', d => console.log(`[UI] ${d}`));
  viteProcess.stderr.on('data', d => console.error(`[UI] ${d}`));
  viteProcess.on('exit', code => console.log(`[UI] vite exited (${code})`));
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 700,
    title: 'Retail AI — Store Intelligence',
    backgroundColor: '#070b12',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });

  try {
    await waitForUrl(VITE_URL, 30000);
    await mainWindow.loadURL(VITE_URL);
  } catch (err) {
    dialog.showErrorBox('Retail AI could not start', `${err.message}\n\nCheck the terminal for backend/frontend errors.`);
  }
}

async function boot() {
  try {
    startBackend();
  } catch (err) {
    dialog.showErrorBox('Retail AI backend error', err.message);
    return;
  }
  startVite();
  await createWindow();
}

function cleanup() {
  for (const child of [backendProcess, viteProcess]) {
    if (!child || child.killed) continue;
    try { child.kill('SIGTERM'); } catch {}
  }
}

app.whenReady().then(boot);
app.on('before-quit', cleanup);
app.on('window-all-closed', () => {
  cleanup();
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
