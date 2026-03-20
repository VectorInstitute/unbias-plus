#!/bin/sh
set -e

# Start the FastAPI/uvicorn server in the background on port 8000.
# nginx (below) proxies API requests to it once it's ready.
unbias-plus --serve --port 8000 &

# Start nginx in the foreground on port 8080.
# nginx serves the UI immediately and proxies /health, /analyze, etc. to uvicorn.
exec nginx -c /app/nginx.conf -g 'daemon off;'
