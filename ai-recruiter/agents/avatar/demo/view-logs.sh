#!/bin/bash
# View and filter avatar demo logs
#
# Usage:
#   ./view-logs.sh                    # All backend logs
#   ./view-logs.sh --avatar           # Avatar agent logs only
#   ./view-logs.sh --viseme           # Viseme extraction logs
#   ./view-logs.sh --sse              # SSE stream logs
#   ./view-logs.sh --scorer           # Scorer pipeline logs
#   ./view-logs.sh --refiner          # Refiner pipeline logs
#   ./view-logs.sh --errors           # Errors only
#   ./view-logs.sh --follow           # Live tail (Ctrl+C to stop)
#   ./view-logs.sh --frontend         # Frontend logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"

FILTER=""
LOG_FILE="$BACKEND_LOG"
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --avatar)    FILTER="avatar";   shift ;;
        --viseme)    FILTER="viseme\|Viseme\|VISEME"; shift ;;
        --sse)       FILTER="SSE\|stream\|event_generator"; shift ;;
        --scorer)    FILTER="SCORER\|scorer"; shift ;;
        --refiner)   FILTER="REFINER\|refiner"; shift ;;
        --errors)    FILTER="ERROR\|CRITICAL\|error\|failed\|Failed"; shift ;;
        --follow|-f) FOLLOW=true; shift ;;
        --frontend)  LOG_FILE="$FRONTEND_LOG"; shift ;;
        --help|-h)
            echo "Usage: $0 [--avatar|--viseme|--sse|--scorer|--refiner|--errors] [--follow] [--frontend]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    echo "Start the demo first with: ./start-avatar-demo.sh"
    exit 1
fi

if [ "$FOLLOW" = true ]; then
    if [ -n "$FILTER" ]; then
        tail -f "$LOG_FILE" | grep --line-buffered -i "$FILTER"
    else
        tail -f "$LOG_FILE"
    fi
else
    if [ -n "$FILTER" ]; then
        grep -i "$FILTER" "$LOG_FILE" | tail -100
    else
        tail -100 "$LOG_FILE"
    fi
fi
