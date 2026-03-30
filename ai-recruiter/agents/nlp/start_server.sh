#!/bin/bash
# ==========================================================================
# NLP Agent Server — Startup Script (Production Edition)
#
# Usage:
#   ./start_server.sh              # Production mode (default)
#   ./start_server.sh --reload     # Development mode (auto-reload)
#   ./start_server.sh --help       # Show help
#
# Environment variables (all optional, have sensible defaults):
#   HOST                  — Bind address (default: 0.0.0.0)
#   PORT                  — Server port (default: 8001)
#   WORKERS               — Uvicorn worker count (default: 1)
#   CUDA_VISIBLE_DEVICES  — GPU to use (default: 0)
#   LOG_LEVEL             — Logging level (default: info)
#   MAX_MODEL_LEN         — vLLM max context length (default: 2048)
#   GPU_MEMORY_UTIL       — GPU memory fraction (default: 0.92)
#   DTYPE                 — Model precision (default: bfloat16)
#   USE_COMPACT_PROMPT    — Use shorter prompt (default: false)
#
# Examples:
#   PORT=8080 ./start_server.sh
#   CUDA_VISIBLE_DEVICES=1 GPU_MEMORY_UTIL=0.85 ./start_server.sh
#   USE_COMPACT_PROMPT=true ./start_server.sh --reload
# ==========================================================================

set -e
# Exit immediately if any command fails.
# This prevents the script from continuing after an error
# (e.g., missing dependency) and silently starting a broken server.

set -u
# Treat unset variables as errors.
# Catches typos in variable names that would otherwise silently expand to "".

set -o pipefail
# If any command in a pipeline fails, the whole pipeline fails.
# Without this, `command1 | command2` only checks command2's exit code.

# ==========================================================================
# Color codes for terminal output
# ==========================================================================

RED='\033[0;31m'
# ANSI escape code for red text. Used for error messages.

GREEN='\033[0;32m'
# Green text. Used for success messages.

YELLOW='\033[0;33m'
# Yellow text. Used for warnings.

BLUE='\033[0;34m'
# Blue text. Used for info messages.

CYAN='\033[0;36m'
# Cyan text. Used for configuration display.

NC='\033[0m'
# "No Color" — resets text color back to default.
# Always use this after a color code to prevent color bleeding.

# ==========================================================================
# Helper functions
# ==========================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    # Print an info message in blue.
    # -e enables escape code interpretation (\033 colors).
    # $1 is the first argument passed to this function.
}

log_ok() {
    echo -e "${GREEN}[  OK]${NC} $1"
    # Print a success message in green.
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    # Print a warning in yellow.
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    # Print an error in red.
}

show_help() {
    # Print usage information and exit.
    echo ""
    echo -e "${CYAN}NLP Agent Server — AI Recruiter Interview Engine${NC}"
    echo ""
    echo "Usage:"
    echo "  ./start_server.sh              # Production mode"
    echo "  ./start_server.sh --reload     # Development mode (auto-reload)"
    echo "  ./start_server.sh --health     # Run health check only"
    echo "  ./start_server.sh --help       # Show this help"
    echo ""
    echo "Environment variables:"
    echo "  HOST                  Bind address          (default: 0.0.0.0)"
    echo "  PORT                  Server port           (default: 8001)"
    echo "  WORKERS               Worker count          (default: 1)"
    echo "  CUDA_VISIBLE_DEVICES  GPU ID                (default: 0)"
    echo "  LOG_LEVEL             Log level             (default: info)"
    echo "  MAX_MODEL_LEN         Max context length    (default: 2048)"
    echo "  GPU_MEMORY_UTIL       GPU memory fraction   (default: 0.92)"
    echo "  DTYPE                 Precision             (default: bfloat16)"
    echo "  USE_COMPACT_PROMPT    Shorter prompt        (default: false)"
    echo ""
    echo "Examples:"
    echo "  PORT=8080 ./start_server.sh"
    echo "  CUDA_VISIBLE_DEVICES=1 GPU_MEMORY_UTIL=0.85 ./start_server.sh"
    echo "  USE_COMPACT_PROMPT=true ./start_server.sh --reload"
    echo ""
    exit 0
    # exit 0 = success. The user asked for help, not an error.
}

# ==========================================================================
# Parse command-line arguments
# ==========================================================================

MODE="production"
# Default to production mode.

RUN_HEALTH_CHECK=false
# Whether to run health check and exit.

# Loop through all arguments passed to the script.
for arg in "$@"; do
    case "$arg" in
        --reload)
            MODE="development"
            # Development mode: uvicorn auto-reloads on file changes.
            ;;
        --health)
            RUN_HEALTH_CHECK=true
            # Just check dependencies and exit.
            ;;
        --help|-h)
            show_help
            # Print help and exit.
            ;;
        *)
            log_error "Unknown argument: $arg"
            echo "Run ./start_server.sh --help for usage."
            exit 1
            # Reject unknown arguments to prevent typos from being silently ignored.
            ;;
    esac
done
# case/esac is bash's version of switch/case.
# Each pattern ends with ;; (like break in other languages).

# ==========================================================================
# Configuration (with defaults)
# ==========================================================================

# Server configuration
HOST="${HOST:-0.0.0.0}"
# ${VAR:-default} means: use $VAR if set and non-empty, otherwise use "default".
# 0.0.0.0 = listen on all network interfaces (accessible from other machines).
# Use 127.0.0.1 to restrict to localhost only.

PORT="${PORT:-8001}"
# Default port 8001.
# Using 8001 instead of 8000 to avoid conflicts with other services.
# The orchestrator will connect to this port.

WORKERS="${WORKERS:-1}"
# Number of uvicorn worker processes.
# IMPORTANT: Must be 1 for vLLM because:
# 1. vLLM loads the model into GPU memory once per process
# 2. Multiple workers = multiple model copies = OOM
# 3. vLLM handles concurrency internally via async
# Only increase if running on CPU-only mode (no vLLM).

GPU="${CUDA_VISIBLE_DEVICES:-0}"
# Which GPU to use. 0 = first GPU.
# Override with CUDA_VISIBLE_DEVICES=1 for the second GPU.

LOG_LEVEL="${LOG_LEVEL:-info}"
# Uvicorn log level: debug, info, warning, error, critical.
# "info" shows request logs. "warning" only shows problems.

# Model configuration
MODEL_PATH="/teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter/ml/recruiter-persona/training/output_v3/recruiter-persona-llama-3.1-8b/merged-full"
# Absolute path to the fine-tuned merged model.
# This path is checked during startup to fail fast if the model is missing.

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
# Maximum context length for vLLM.
# 2048 is tight but sufficient for interviews:
# System prompt (~600) + history (~600) + current (~200) + generation (~48) ≈ 1448

GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
# Fraction of GPU memory vLLM can use.
# 0.92 = use 92%, leave 8% for PyTorch overhead.
# Lower this if you see CUDA OOM errors.

DTYPE="${DTYPE:-bfloat16}"
# Model precision. bfloat16 preferred for LLaMA 3.1.
# float16 is an alternative if bfloat16 isn't supported by the GPU.

USE_COMPACT_PROMPT="${USE_COMPACT_PROMPT:-false}"
# Whether to use the shorter system prompt.
# true = ~150 tokens (faster TTFT)
# false = ~600 tokens (better behavior control)

# ==========================================================================
# Export environment variables for the Python server
# ==========================================================================

export CUDA_VISIBLE_DEVICES="$GPU"
# Tell PyTorch/CUDA which GPU to use.
# This MUST be set before importing torch, which happens when
# the Python server starts.

export NLP_MODEL_PATH="$MODEL_PATH"
# Pass model path to the Python server via environment variable.
# The server reads this with os.getenv("NLP_MODEL_PATH", default_path).

export NLP_MAX_MODEL_LEN="$MAX_MODEL_LEN"
# Pass max model length to the server.

export NLP_GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL"
# Pass GPU memory utilization to the server.

export NLP_DTYPE="$DTYPE"
# Pass model precision to the server.

export NLP_USE_COMPACT_PROMPT="$USE_COMPACT_PROMPT"
# Pass prompt mode to the server.

export NLP_PORT="$PORT"
# Pass port to the server (for self-registration with the orchestrator).

# ==========================================================================
# Dependency checks
# ==========================================================================

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  NLP Agent Server — Startup Checks${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# --- Check Python version ---
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
# Run `python3 --version` and extract the version number.
# awk '{print $2}' extracts the second word: "Python 3.12.0" → "3.12.0"

PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
# Extract major version: "3.12.0" → "3"

PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
# Extract minor version: "3.12.0" → "12"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    log_error "Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
# vLLM requires Python 3.10+. Check and fail fast if too old.

log_ok "Python $PYTHON_VERSION"

# --- Check required Python packages ---
REQUIRED_PACKAGES=("vllm" "transformers" "uvicorn" "fastapi")
# List of packages the server needs.
# We check each one individually to give specific error messages.

MISSING_PACKAGES=()
# Will hold the names of any missing packages.

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        log_ok "$pkg installed"
    else
        log_error "$pkg NOT installed"
        MISSING_PACKAGES+=("$pkg")
        # Add to the missing list.
    fi
done
# Loop through each package and try to import it.
# 2>/dev/null suppresses error output from failed imports.

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    # ${#array[@]} = length of the array.
    echo ""
    log_error "Missing packages: ${MISSING_PACKAGES[*]}"
    # ${array[*]} joins all elements with spaces.
    echo ""
    echo "Install with:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Or individually:"
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "  pip install $pkg"
    done
    exit 1
fi

# --- Check GPU availability ---
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    # Get the GPU name for display.

    GPU_MEM=$(python3 -c "
import torch
mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
print(f'{mem:.1f}')
" 2>/dev/null)
    # Get total GPU memory in GB.
    # total_mem is in bytes, so divide by 1024^3 to get GB.

    log_ok "GPU: $GPU_NAME (${GPU_MEM}GB)"
else
    log_warn "No GPU detected! vLLM will be very slow on CPU."
    echo "  If you have a GPU, check CUDA installation:"
    echo "  python3 -c \"import torch; print(torch.cuda.is_available())\""
fi

# --- Check model directory ---
if [ -d "$MODEL_PATH" ]; then
    # -d checks if the path exists and is a directory.

    # Count model shard files
    SHARD_COUNT=$(find "$MODEL_PATH" -name "model-*.safetensors" 2>/dev/null | wc -l)
    # find searches for files matching the pattern.
    # wc -l counts the number of lines (= number of files found).

    CONFIG_EXISTS=$([ -f "$MODEL_PATH/config.json" ] && echo "yes" || echo "no")
    # -f checks if the path exists and is a regular file.
    # && echo "yes" || echo "no" is a ternary-like pattern in bash.

    TOKENIZER_EXISTS=$([ -f "$MODEL_PATH/tokenizer.json" ] && echo "yes" || echo "no")
    # Check if the tokenizer file exists.

    if [ "$CONFIG_EXISTS" = "yes" ] && [ "$TOKENIZER_EXISTS" = "yes" ]; then
        log_ok "Model found: $MODEL_PATH"
        log_ok "  Shards: $SHARD_COUNT | Config: $CONFIG_EXISTS | Tokenizer: $TOKENIZER_EXISTS"
    else
        log_warn "Model directory exists but may be incomplete:"
        log_warn "  Config: $CONFIG_EXISTS | Tokenizer: $TOKENIZER_EXISTS"
        echo "  Expected files: config.json, tokenizer.json, model-*.safetensors"
    fi
else
    log_error "Model directory NOT found:"
    echo "  $MODEL_PATH"
    echo ""
    echo "  Please verify the model path."
    echo "  If the model hasn't been merged yet, run:"
    echo "    python scripts/merge_adapters.py"
    exit 1
fi

# --- Check server source files ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the directory where THIS script is located.
# ${BASH_SOURCE[0]} = path to this script.
# dirname = extract directory part.
# cd + pwd = resolve to absolute path (handles symlinks).

SERVER_FILES=("server.py" "agent.py" "vllm_engine.py" "interview_state.py")
# List of Python files the server needs.

for f in "${SERVER_FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        log_ok "$f found"
    else
        log_error "$f NOT found in $SCRIPT_DIR"
        exit 1
    fi
done

# ==========================================================================
# Health check mode
# ==========================================================================

if [ "$RUN_HEALTH_CHECK" = true ]; then
    echo ""
    log_ok "All checks passed! Server is ready to start."
    echo ""
    echo "Run without --health to start the server."
    exit 0
fi
# If --health flag was passed, just run checks and exit.
# Useful for CI/CD pipelines or Docker health checks.

# ==========================================================================
# Display configuration
# ==========================================================================

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Configuration${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "  ${CYAN}Mode:${NC}              $MODE"
echo -e "  ${CYAN}Host:${NC}              $HOST"
echo -e "  ${CYAN}Port:${NC}              $PORT"
echo -e "  ${CYAN}Workers:${NC}           $WORKERS"
echo -e "  ${CYAN}GPU:${NC}               $GPU"
echo -e "  ${CYAN}Log Level:${NC}         $LOG_LEVEL"
echo -e "  ${CYAN}Max Model Len:${NC}     $MAX_MODEL_LEN"
echo -e "  ${CYAN}GPU Memory Util:${NC}   $GPU_MEMORY_UTIL"
echo -e "  ${CYAN}Dtype:${NC}             $DTYPE"
echo -e "  ${CYAN}Compact Prompt:${NC}    $USE_COMPACT_PROMPT"
echo -e "  ${CYAN}Model:${NC}             $(basename $MODEL_PATH)"
# basename extracts the last component of the path:
# "/long/path/to/merged-full" → "merged-full"

echo ""

# ==========================================================================
# Workers safety check
# ==========================================================================

if [ "$WORKERS" -gt 1 ]; then
    log_warn "Workers > 1 detected ($WORKERS)."
    log_warn "vLLM loads the full model per worker."
    log_warn "This WILL cause GPU OOM unless you have ${WORKERS}x GPU memory."
    echo ""
    echo -e "  Recommended: ${CYAN}WORKERS=1${NC} (vLLM handles concurrency internally)"
    echo ""
    read -p "  Continue anyway? [y/N] " -n 1 -r
    # read -p = prompt, -n 1 = read 1 character, -r = raw (no backslash escaping)
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        # =~ is regex match. ^[Yy]$ matches "y" or "Y".
        echo "Aborted."
        exit 1
    fi
fi

# ==========================================================================
# Pre-start: change to script directory
# ==========================================================================

cd "$SCRIPT_DIR"
# Change to the directory where the Python files are.
# This ensures `uvicorn server:app` can find server.py
# regardless of where the script was called from.

log_info "Working directory: $(pwd)"

# ==========================================================================
# Start the server
# ==========================================================================

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Starting Server${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [ "$MODE" = "development" ]; then
    log_info "Running in DEVELOPMENT mode (auto-reload enabled)"
    log_warn "Auto-reload will restart the server on file changes."
    log_warn "Model will be reloaded on each restart (~60 seconds)."
    echo ""

    uvicorn server:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --reload \
        --reload-dir "$SCRIPT_DIR"
    # --reload: Watch for file changes and restart automatically.
    #   Great for development, but every restart reloads the model.
    # --reload-dir: Only watch THIS directory for changes.
    #   Prevents reloads from unrelated file changes elsewhere.
    # Note: No --workers flag because --reload is incompatible with multiple workers.

else
    log_info "Running in PRODUCTION mode"
    echo ""

    # Trap SIGTERM and SIGINT for graceful shutdown
    trap 'echo ""; log_info "Shutting down gracefully..."; kill -- -$$; exit 0' SIGTERM SIGINT
    # trap catches signals and runs the specified command.
    # SIGTERM = sent by `kill <pid>` or Docker stop.
    # SIGINT = sent by Ctrl+C.
    # kill -- -$$ = send signal to the entire process group.
    # $$ = PID of this script. -$$ = negative PID = process group.
    # This ensures child processes (uvicorn) are also terminated.

    uvicorn server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --timeout-keep-alive 120 \
        --limit-concurrency 100
    # --workers: Number of worker processes (should be 1 for vLLM).
    # --timeout-keep-alive 120: Keep HTTP connections alive for 2 minutes.
    #   This prevents reconnection overhead for the orchestrator
    #   which sends requests frequently.
    # --limit-concurrency 100: Max simultaneous connections.
    #   Prevents the server from being overwhelmed.
    #   Excess connections get HTTP 503 (Service Unavailable).
fi