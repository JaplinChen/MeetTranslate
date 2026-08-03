@echo off
setlocal
cd /d "%~dp0"
set PORT=8010

rem Free the port first so the script is re-runnable without manual cleanup.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"LISTENING .*:%PORT% " 2^>nul') do (
    echo Stopping process %%p on port %PORT%
    taskkill /F /PID %%p >nul 2>&1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :fail
    .venv\Scripts\python.exe -m pip install -q -r requirements.txt || goto :fail

    rem A machine with an NVIDIA card runs recognition nine times faster on it. Installed here
    rem rather than left to the README, because the CPU fallback is silent — you would never
    rem know the card was idle.
    nvidia-smi >nul 2>&1 && (
        echo NVIDIA GPU detected, installing GPU recognition...
        .venv\Scripts\python.exe -m pip install -q -r requirements-gpu.txt
    )
)

if not exist "dashboard\dist\index.html" (
    echo Building dashboard...
    pushd dashboard && call npm install --silent && call npm run build && popd || goto :fail
)

start "" http://127.0.0.1:%PORT%/
.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port %PORT%
goto :eof

:fail
echo.
echo Startup failed. See the messages above.
pause
exit /b 1
