# Demo Logging - Quick Start

## ✅ What Was Added

The React demo startup script now includes **comprehensive logging**!

### Modified Files
- `demo/start-react-demo.sh` - Now saves logs to files

### New Files
- `demo/view-logs.sh` - Easy log viewer with filters
- `demo/DEMO_LOGGING.md` - Complete documentation

---

## 🚀 Quick Start

### 1. Start the Demo
```bash
cd ai-recruiter/agents/nlp/demo
./start-react-demo.sh
```

You'll see:
```
========================================
  ✅ Both servers running!
========================================

  🌐 Access the app:
     http://localhost:3000

  📊 View pipeline logs:
     tail -f backend.log

  🔍 Search logs:
     grep 'SCORER' backend.log
     grep 'REFINER' backend.log
     grep 'FINAL' backend.log
```

### 2. View Logs (Pick One)

**Option A: Use the log viewer (easiest)**
```bash
./view-logs.sh -f           # All logs, real-time
./view-logs.sh --scorer -f  # Scorer only
./view-logs.sh --refiner -f # Refiner only
./view-logs.sh --final      # Final questions
```

**Option B: Direct commands**
```bash
tail -f backend.log              # All logs
grep "SCORER" backend.log        # Scorer analysis
grep "REFINER" backend.log       # Refinement
grep "FINAL" backend.log         # Final questions
```

### 3. Run an Interview

Open http://localhost:3000 and start asking/answering questions. Watch the logs to see the full pipeline in action!

---

## 📊 What You'll See

```
======================================================================
📊 PIPELINE LOG - Session: a1b2c3d4
======================================================================

🧠 SCORER - Analyzing candidate answer...
   ✓ Scorer completed in 450ms
   • Answer Quality: detailed
   • Acknowledgment: "So you used Neo4j for the graph layer."

📋 DECISION - Move to new topic

🤖 MAIN LLM - Generating question...
   📝 Original: "Tell me about your React frontend."

✨ REFINER - Enhancing question...
   ✏️ Refined: "What specific challenges did you face with React 
               state management when scaling to thousands of users?"

✅ FINAL: "So you used Neo4j for the graph layer. What specific 
          challenges did you face with React state management?"
======================================================================
```

---

## 💡 Pro Tips

### Multi-Terminal Setup
```bash
# Terminal 1: Run demo
./start-react-demo.sh

# Terminal 2: Watch all logs
./view-logs.sh -f

# Terminal 3: Monitor specific component
./view-logs.sh --refiner -f
```

### Quick Checks
```bash
# See all final questions
./view-logs.sh --final

# Count refinement types
grep "Type:" backend.log | cut -d: -f3 | sort | uniq -c

# Check for errors
grep "✗\|⚠" backend.log
```

---

## 📁 Log Files

- `backend.log` - Pipeline logs (scorer, LLM, refiner)
- `frontend.log` - React dev server logs

Logs are **cleared on each startup** automatically.

---

## 🎯 Key Features

✅ Pipeline logs saved to files  
✅ Easy log viewer with filters  
✅ Real-time following  
✅ Search by component  
✅ Instructions displayed on startup  
✅ Automatic log cleanup  

---

## 🔧 Troubleshooting

**No logs?**
- Check demo is running: `ps aux | grep uvicorn`
- Check files exist: `ls -l backend.log`

**Logs empty?**
- Demo might still be starting
- Try after making a request in the UI

**Want more details?**
- See `demo/DEMO_LOGGING.md` for complete guide

---

That's it! Your demo now has full visibility into the 3-LLM pipeline. 🎉

Happy interviewing! 🚀
