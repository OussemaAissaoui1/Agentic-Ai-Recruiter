#!/bin/bash
# AI Recruiter Demo — Production Startup Script
#
# Usage: ./start.sh [--cleanup] [--port PORT]
#
# Options:
#   --cleanup    Kill existing vLLM processes before starting
#   --port PORT  Streamlit port (default: 8501)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PORT=8501
CLEANUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  AI Recruiter Demo — Startup"
echo "=========================================="
echo ""

# Check Python
echo "[1/5] Checking Python..."
python3 --version || { echo "ERROR: Python3 not found"; exit 1; }

# Check GPU
echo ""
echo "[2/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv

    # Get free memory
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo ""
    echo "    Free GPU memory: ${FREE_MEM} MiB"

    # Check if enough memory
    if [ "$FREE_MEM" -lt 12000 ]; then
        echo "    WARNING: Less than 12GB GPU memory free"

        if [ "$CLEANUP" = true ]; then
            echo "    Cleaning up GPU processes..."
            # Kill vLLM processes
            nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
                if [ -n "$pid" ]; then
                    CMD=$(ps -p "$pid" -o comm= 2>/dev/null || echo "")
                    if [[ "$CMD" == *"python"* ]] || [[ "$CMD" == *"vllm"* ]]; then
                        echo "    Killing PID $pid ($CMD)"
                        kill -9 "$pid" 2>/dev/null || true
                    fi
                fi
            done
            sleep 2
            echo "    Cleanup complete"
        else
            echo "    Run with --cleanup to kill existing processes"
        fi
    else
        echo "    GPU memory OK"
    fi
else
    echo "    WARNING: nvidia-smi not found"
fi

# Check if port is available
echo ""
echo "[3/5] Checking port $PORT..."
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "    WARNING: Port $PORT is in use"
    echo "    Processes using port $PORT:"
    lsof -i :$PORT || true

    read -p "    Kill processes on port $PORT? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti :$PORT | xargs kill -9 2>/dev/null || true
        sleep 1
        echo "    Port freed"
    fi
else
    echo "    Port $PORT is available"
fi

# Check model path
echo ""
echo "[4/5] Checking model..."
MODEL_PATH="${NLP_MODEL_PATH:-/teamspace/studios/this_studio/Agentic-Ai-Recruiter/model_cache}"
if [ -d "$MODEL_PATH" ]; then
    SHARDS=$(ls -1 "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
    echo "    Model found: $SHARDS safetensors shards"
else
    echo "    ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Start Streamlit
echo ""
echo "[5/5] Starting Streamlit on port $PORT..."
echo ""
echo "=========================================="
echo "  Access the demo at: http://localhost:$PORT"
echo "=========================================="
echo ""

exec streamlit run demo/app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
