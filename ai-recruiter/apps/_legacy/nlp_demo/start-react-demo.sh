#!/bin/bash
# AI Recruiter React Demo — Startup Script for Lightning AI
#
# This script starts both the FastAPI backend and Vite dev server
# Logs are saved to: backend.log and frontend.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log files
BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"

echo "=========================================="
echo "  AI Recruiter React Demo — Startup"
echo "=========================================="
echo ""

# Default ports
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Parse arguments
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--backend-port PORT] [--frontend-port PORT]"
            exit 1
            ;;
    esac
done

echo "[1/3] Checking dependencies..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Please install Node.js"
    exit 1
fi
echo "    Node.js: $(node --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi
echo "    Python: $(python3 --version)"

# Check if npm packages are installed
if [ ! -d "node_modules" ]; then
    echo ""
    echo "[2/3] Installing npm dependencies..."
    npm install
else
    echo ""
    echo "[2/3] npm dependencies already installed"
fi

echo ""
echo "[3/3] Starting servers..."
echo ""
echo "Backend (FastAPI):  http://localhost:$BACKEND_PORT"
echo "Frontend (React):   http://localhost:$FRONTEND_PORT"
echo ""
echo "📊 PIPELINE LOGS:"
echo "   Backend:  $BACKEND_LOG"
echo "   Frontend: $FRONTEND_LOG"
echo ""
echo "View logs in real-time:"
echo "   tail -f $BACKEND_LOG"
echo ""
echo "=========================================="

# Trap to kill both processes on Ctrl+C
trap 'echo ""; echo "Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT TERM

# Clear old logs
> "$BACKEND_LOG"
> "$FRONTEND_LOG"

# Start backend in background with logging
echo "Starting FastAPI backend on port $BACKEND_PORT..."
echo "Logs: $BACKEND_LOG"
cd "$SCRIPT_DIR/.."
python3 -m uvicorn demo.server:app --host 0.0.0.0 --port $BACKEND_PORT > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend with logging
echo "Starting Vite dev server on port $FRONTEND_PORT..."
echo "Logs: $FRONTEND_LOG"
cd "$SCRIPT_DIR"
BACKEND_URL="http://localhost:$BACKEND_PORT" npm run dev -- --port $FRONTEND_PORT > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "  ✅ Both servers running!"
echo "=========================================="
echo ""
echo "  🌐 Access the app:"
echo "     http://localhost:$FRONTEND_PORT"
echo ""
echo "  📊 View pipeline logs:"
echo "     tail -f $BACKEND_LOG"
echo ""
echo "  🔍 Search logs:"
echo "     grep 'SCORER' $BACKEND_LOG      # Scorer analysis"
echo "     grep 'REFINER' $BACKEND_LOG     # Question refinement"
echo "     grep 'FINAL' $BACKEND_LOG       # Final questions"
echo ""
echo "  ⏹  Stop servers: Press Ctrl+C"
echo "=========================================="

# Wait for both processes
wait
