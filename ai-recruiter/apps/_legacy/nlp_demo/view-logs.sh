#!/bin/bash
# AI Recruiter Demo — Log Viewer
# Displays the pipeline logs in real-time with optional filtering

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_LOG="$SCRIPT_DIR/backend.log"

# Check if log file exists
if [ ! -f "$BACKEND_LOG" ]; then
    echo "❌ Backend log file not found: $BACKEND_LOG"
    echo ""
    echo "Make sure the demo is running:"
    echo "  ./start-react-demo.sh"
    exit 1
fi

# Parse arguments
FILTER=""
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --scorer)
            FILTER="SCORER"
            shift
            ;;
        --refiner)
            FILTER="REFINER"
            shift
            ;;
        --final)
            FILTER="FINAL"
            shift
            ;;
        --decision)
            FILTER="DECISION"
            shift
            ;;
        --llm)
            FILTER="MAIN LLM"
            shift
            ;;
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "View AI Recruiter pipeline logs"
            echo ""
            echo "Options:"
            echo "  --scorer      Show only scorer analysis"
            echo "  --refiner     Show only refiner output"
            echo "  --final       Show only final questions"
            echo "  --decision    Show only decision making"
            echo "  --llm         Show only main LLM output"
            echo "  -f, --follow  Follow log in real-time (like tail -f)"
            echo "  -h, --help    Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                    # Show all logs"
            echo "  $0 --scorer           # Show scorer analysis only"
            echo "  $0 --refiner -f       # Follow refiner output in real-time"
            echo "  $0 --final            # Show final questions only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  AI Recruiter Pipeline Logs"
echo "=========================================="
echo ""
echo "Log file: $BACKEND_LOG"
if [ -n "$FILTER" ]; then
    echo "Filter: $FILTER"
fi
if [ "$FOLLOW" = true ]; then
    echo "Mode: Follow (real-time)"
else
    echo "Mode: Static"
fi
echo ""
echo "=========================================="
echo ""

# Display logs
if [ "$FOLLOW" = true ]; then
    if [ -n "$FILTER" ]; then
        tail -f "$BACKEND_LOG" | grep --line-buffered "$FILTER"
    else
        tail -f "$BACKEND_LOG"
    fi
else
    if [ -n "$FILTER" ]; then
        grep "$FILTER" "$BACKEND_LOG"
    else
        cat "$BACKEND_LOG"
    fi
fi
