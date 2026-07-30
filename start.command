#!/bin/bash
set -e
cd "$(dirname "$0")"
PORT=8000

# Free the port first so the script is re-runnable without manual cleanup.
if pid=$(lsof -ti tcp:$PORT 2>/dev/null); then
    echo "Stopping process $pid on port $PORT"
    kill -9 $pid 2>/dev/null || true
fi

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/python -m pip install -q -r requirements.txt
fi

if [ ! -f dashboard/dist/index.html ]; then
    echo "Building dashboard..."
    (cd dashboard && npm install --silent && npm run build)
fi

open "http://127.0.0.1:$PORT/"
exec .venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port $PORT
