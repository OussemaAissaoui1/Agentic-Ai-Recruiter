#!/bin/bash
# ==========================================================================
#  AI Recruiter Avatar Demo — Startup Script
#
#  Starts the unified FastAPI backend (NLP + Avatar) and the Vite
#  React frontend with Three.js avatar rendering.
#
#  Usage:
#    ./start-avatar-demo.sh                           # Default ports
#    ./start-avatar-demo.sh --backend-port 8001       # Custom backend port
#    ./start-avatar-demo.sh --frontend-port 3001      # Custom frontend port
#    ./start-avatar-demo.sh --dev                     # Dev mode with reload
#
#  Logs:
#    backend.log   — FastAPI + NLP agent + Avatar agent logs
#    frontend.log  — Vite dev server logs
#
#  Architecture:
#    Browser (Three.js + React)
#      ├── /api/*       ──► FastAPI (NLP pipeline)
#      └── /avatar/*    ──► FastAPI (GLB serving + viseme extraction)
# ==========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVATAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NLP_DIR="$(cd "$AVATAR_DIR/../nlp" && pwd)"
AGENTS_DIR="$(cd "$AVATAR_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

# Log files
BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BACKEND_PORT=8001
FRONTEND_PORT=3001
DEV_MODE=false

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
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend-port PORT   Backend port (default: 8001)"
            echo "  --frontend-port PORT  Frontend port (default: 3001)"
            echo "  --dev                 Enable hot-reload for backend"
            echo "  --help                Show this help"
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
echo "╔══════════════════════════════════════════╗"
echo "║   AI Recruiter Avatar Demo — Startup    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Backend (FastAPI):  http://localhost:$BACKEND_PORT"
echo "  Frontend (React):   http://localhost:$FRONTEND_PORT"
echo "  Dev mode:           $DEV_MODE"
echo ""

# ---------------------------------------------------------------------------
# [1/4] Check dependencies
# ---------------------------------------------------------------------------
echo "[1/4] Checking dependencies..."

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
for pkg in fastapi uvicorn numpy; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "  Installing missing Python packages: ${MISSING_PKGS[*]}"
    pip install -q "${MISSING_PKGS[@]}"
fi
echo "  Python packages: OK"

# GLB avatar file
GLB_FILE="$AVATAR_DIR/ready_player_me_female_avatar__vrchatgame.glb"
if [ ! -f "$GLB_FILE" ]; then
    echo "  ERROR: GLB avatar file not found: $GLB_FILE"
    echo "  Place your Ready Player Me .glb file at the path above"
    exit 1
fi
GLB_SIZE=$(du -h "$GLB_FILE" | cut -f1)
echo "  Avatar GLB: $GLB_SIZE"

# GPU check (non-fatal)
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  GPU free: ${GPU_MEM}MB"
else
    echo "  GPU: nvidia-smi not found (CPU mode)"
fi

echo ""

# ---------------------------------------------------------------------------
# [2/4] Install npm dependencies
# ---------------------------------------------------------------------------
echo "[2/4] Checking npm dependencies..."

if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo "  Installing npm packages (first run)..."
    npm install --prefer-offline 2>&1 | tail -3
else
    echo "  node_modules/ exists — skipping install"
    echo "  (Run 'npm install' manually to update)"
fi

echo ""

# ---------------------------------------------------------------------------
# [3/4] Start backend
# ---------------------------------------------------------------------------
echo "[3/4] Starting FastAPI backend on port $BACKEND_PORT..."

# Clear old logs
> "$BACKEND_LOG"
> "$FRONTEND_LOG"

# Build PYTHONPATH so imports resolve correctly
export PYTHONPATH="$NLP_DIR:$AGENTS_DIR:${PYTHONPATH:-}"

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

# Wait for backend to start (up to 120s for model loading)
echo "  Waiting for backend..."
WAIT_MAX=120
WAIT_ELAPSED=0
while [ $WAIT_ELAPSED -lt $WAIT_MAX ]; do
    if curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        echo "  Backend is up! (${WAIT_ELAPSED}s)"
        break
    fi

    # Check if process died
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "  ERROR: Backend process died. Check logs:"
        echo "  tail -50 $BACKEND_LOG"
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
    echo "  It may still be loading models. Check logs:"
    echo "  tail -f $BACKEND_LOG"
fi

echo ""

# ---------------------------------------------------------------------------
# [4/4] Start frontend
# ---------------------------------------------------------------------------
echo "[4/4] Starting Vite dev server on port $FRONTEND_PORT..."

cd "$SCRIPT_DIR"
BACKEND_URL="http://localhost:$BACKEND_PORT" \
    npx vite --host --port "$FRONTEND_PORT" > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

echo "  PID: $FRONTEND_PID"
echo "  Logs: $FRONTEND_LOG"

# Brief wait for Vite to start
sleep 3

echo ""

# ---------------------------------------------------------------------------
# Ready
# ---------------------------------------------------------------------------
echo "╔══════════════════════════════════════════╗"
echo "║        Both servers running!             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Open in browser:"
echo "    http://localhost:$FRONTEND_PORT"
echo ""
echo "  View backend logs (NLP + Avatar pipeline):"
echo "    tail -f $BACKEND_LOG"
echo ""
echo "  View frontend logs:"
echo "    tail -f $FRONTEND_LOG"
echo ""
echo "  Search logs:"
echo "    grep 'SCORER'  $BACKEND_LOG     # Answer scoring"
echo "    grep 'REFINER' $BACKEND_LOG     # Question refinement"
echo "    grep 'viseme'  $BACKEND_LOG     # Lip sync extraction"
echo "    grep 'avatar'  $BACKEND_LOG     # Avatar agent"
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
    # Give processes time to clean up
    sleep 1
    # Force kill if still running
    kill -9 $BACKEND_PID 2>/dev/null
    kill -9 $FRONTEND_PID 2>/dev/null
    echo "Done."
    exit 0
}

trap cleanup INT TERM

# Wait for both processes
wait
