@echo off
setlocal
cd /d "%~dp0"
set "VAS_APP=%~dp0src-tauri\target\release\vision-analytics-studio.exe"

if not exist "%VAS_APP%" (
  echo Vision Analytics Studio Release version is not built yet.
  echo Run: npm run desktop:build
  pause
  exit /b 1
)

start "" "%VAS_APP%"
