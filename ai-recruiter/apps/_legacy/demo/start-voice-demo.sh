#!/bin/bash
# ==========================================================================
#  AI Recruiter Voice Demo — Startup Script
#
#  Starts the unified FastAPI backend (NLP + Avatar + Faster Whisper STT)
#  and the Vite React frontend with voice input + 3D avatar.
#
#  Usage:
#    ./start-voice-demo.sh                           # Default ports
#    ./start-voice-demo.sh --backend-port 8002       # Custom backend port
#    ./start-voice-demo.sh --frontend-port 3002      # Custom frontend port
#    ./start-voice-demo.sh --whisper-model medium     # Whisper model size
#    ./start-voice-demo.sh --dev                     # Dev mode with reload
#
#  Logs:
#    backend.log   — FastAPI + NLP + Avatar + STT logs
#    frontend.log  — Vite dev server logs
# ==========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENTS_DIR="$(cd "$VOICE_DIR/.." && pwd)"
AVATAR_DIR="$AGENTS_DIR/avatar"
NLP_DIR="$AGENTS_DIR/nlp"

cd "$SCRIPT_DIR"

BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BACKEND_PORT=8002
FRONTEND_PORT=3002
DEV_MODE=false
WHISPER_MODEL="small"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend-port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --frontend-port)
            FRONTEND_PORT="$2"
            shift 2
            ;;
        --whisper-model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend-port PORT     Backend port (default: 8002)"
            echo "  --frontend-port PORT    Frontend port (default: 3002)"
            echo "  --whisper-model MODEL   Whisper model size: tiny/base/small/medium/large (default: small)"
            echo "  --dev                   Enable hot-reload for backend"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   AI Recruiter Voice Demo — Startup         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Backend (FastAPI):  http://localhost:$BACKEND_PORT"
echo "  Frontend (React):   http://localhost:$FRONTEND_PORT"
echo "  Whisper model:      $WHISPER_MODEL"
echo "  Dev mode:           $DEV_MODE"
echo ""

# ---------------------------------------------------------------------------
# [1/5] Check dependencies
# ---------------------------------------------------------------------------
echo "[1/5] Checking dependencies..."

# Python
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: Python 3 not found"
    exit 1
fi
PYTHON_VER=$(python3 --version 2>&1)
echo "  Python:  $PYTHON_VER"

# Node.js
if ! command -v node &> /dev/null; then
    echo "  ERROR: Node.js not found. Install with: apt install nodejs"
    exit 1
fi
echo "  Node.js: $(node --version)"

# Python packages
echo "  Checking Python packages..."
MISSING_PKGS=()
for pkg in fastapi uvicorn numpy faster_whisper; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "  Installing missing Python packages: ${MISSING_PKGS[*]}"
    pip install -q faster-whisper fastapi uvicorn numpy python-multipart
fi
echo "  Python packages: OK"

# GLB avatar file
GLB_FILE="$AVATAR_DIR/brunette.glb"
if [ ! -f "$GLB_FILE" ]; then
    echo "  WARNING: GLB avatar file not found: $GLB_FILE"
    echo "  Avatar will not render. Place your .glb file at the path above."
fi

# GPU check
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  GPU free: ${GPU_MEM}MB"
else
    echo "  GPU: nvidia-smi not found (CPU mode — Whisper will be slower)"
fi

echo ""

# ---------------------------------------------------------------------------
# [2/5] Install npm dependencies
# ---------------------------------------------------------------------------
echo "[2/5] Checking npm dependencies..."

if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo "  Installing npm packages (first run)..."
    npm install --prefer-offline 2>&1 | tail -3
else
    echo "  node_modules/ exists — skipping install"
fi

echo ""

# ---------------------------------------------------------------------------
# [3/5] Start backend
# ---------------------------------------------------------------------------
echo "[3/5] Starting FastAPI backend on port $BACKEND_PORT..."

> "$BACKEND_LOG"
> "$FRONTEND_LOG"

export PYTHONPATH="$NLP_DIR:$AGENTS_DIR:${PYTHONPATH:-}"
export WHISPER_MODEL="$WHISPER_MODEL"

RELOAD_FLAG=""
if [ "$DEV_MODE" = true ]; then
    RELOAD_FLAG="--reload"
fi

cd "$SCRIPT_DIR"
python3 -m uvicorn server:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --log-level info \
    $RELOAD_FLAG \
    > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo "  PID: $BACKEND_PID"
echo "  Logs: $BACKEND_LOG"

# Wait for backend
echo "  Waiting for backend (loading NLP + Avatar + Whisper models)..."
WAIT_MAX=180
WAIT_ELAPSED=0
while [ $WAIT_ELAPSED -lt $WAIT_MAX ]; do
    if curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        echo "  Backend is up! (${WAIT_ELAPSED}s)"
        break
    fi

    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "  ERROR: Backend process died. Check logs:"
        tail -20 "$BACKEND_LOG"
        exit 1
    fi

    sleep 3
    WAIT_ELAPSED=$((WAIT_ELAPSED + 3))
    if [ $((WAIT_ELAPSED % 15)) -eq 0 ]; then
        echo "  Still loading models... (${WAIT_ELAPSED}s)"
    fi
done

if [ $WAIT_ELAPSED -ge $WAIT_MAX ]; then
    echo "  WARNING: Backend didn't respond within ${WAIT_MAX}s"
    echo "  It may still be loading. Check: tail -f $BACKEND_LOG"
fi

echo ""

# ---------------------------------------------------------------------------
# [4/5] Start frontend
# ---------------------------------------------------------------------------
echo "[4/5] Starting Vite dev server on port $FRONTEND_PORT..."

cd "$SCRIPT_DIR"
BACKEND_URL="http://localhost:$BACKEND_PORT" \
    npx vite --host --port "$FRONTEND_PORT" > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

echo "  PID: $FRONTEND_PID"
echo "  Logs: $FRONTEND_LOG"

sleep 3

echo ""

# ---------------------------------------------------------------------------
# [5/5] Ready
# ---------------------------------------------------------------------------
echo "╔══════════════════════════════════════════════╗"
echo "║        All servers running!                  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Open in browser:"
echo "    http://localhost:$FRONTEND_PORT"
echo ""
echo "  Features:"
echo "    - 3D Avatar with lip sync"
echo "    - Real-time voice input (Faster Whisper STT)"
echo "    - Live transcript in chat panel"
echo "    - Text input option"
echo ""
echo "  View logs:"
echo "    tail -f $BACKEND_LOG      # Backend (NLP + Avatar + STT)"
echo "    tail -f $FRONTEND_LOG     # Frontend (Vite)"
echo ""
echo "  Search logs:"
echo "    grep 'Whisper'  $BACKEND_LOG     # STT transcription"
echo "    grep 'WS'       $BACKEND_LOG     # WebSocket events"
echo "    grep 'SCORER'   $BACKEND_LOG     # Answer scoring"
echo "    grep 'viseme'   $BACKEND_LOG     # Lip sync"
echo ""
echo "  Stop: Press Ctrl+C"
echo ""

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null && echo "  Backend stopped (PID $BACKEND_PID)"
    kill $FRONTEND_PID 2>/dev/null && echo "  Frontend stopped (PID $FRONTEND_PID)"
    sleep 1
    kill -9 $BACKEND_PID 2>/dev/null
    kill -9 $FRONTEND_PID 2>/dev/null
    echo "Done."
    exit 0
}

trap cleanup INT TERM

wait
