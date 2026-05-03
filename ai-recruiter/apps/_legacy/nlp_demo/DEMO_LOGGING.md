# Demo Logging Guide

## 📊 Viewing Pipeline Logs

When you run the React demo, all pipeline logs are saved to files for easy viewing.

### Starting the Demo

```bash
cd ai-recruiter/agents/nlp/demo
./start-react-demo.sh
```

You'll see:
```
========================================
  AI Recruiter React Demo — Startup
========================================

[3/3] Starting servers...

Backend (FastAPI):  http://localhost:8000
Frontend (React):   http://localhost:3000

📊 PIPELINE LOGS:
   Backend:  /path/to/demo/backend.log
   Frontend: /path/to/demo/frontend.log

View logs in real-time:
   tail -f /path/to/demo/backend.log

========================================
  ✅ Both servers running!
========================================

  🌐 Access the app:
     http://localhost:3000

  📊 View pipeline logs:
     tail -f /path/to/demo/backend.log

  🔍 Search logs:
     grep 'SCORER' backend.log      # Scorer analysis
     grep 'REFINER' backend.log     # Question refinement
     grep 'FINAL' backend.log       # Final questions

  ⏹  Stop servers: Press Ctrl+C
========================================
```

---

## 🔍 Viewing Logs

### Method 1: Use the Log Viewer Script (Easiest)

```bash
cd ai-recruiter/agents/nlp/demo

# View all logs
./view-logs.sh

# View specific components
./view-logs.sh --scorer       # Scorer analysis only
./view-logs.sh --refiner      # Refiner output only
./view-logs.sh --final        # Final questions only
./view-logs.sh --decision     # Decision making only
./view-logs.sh --llm          # Main LLM output only

# Follow logs in real-time
./view-logs.sh --follow       # All logs
./view-logs.sh --scorer -f    # Scorer only, real-time
./view-logs.sh --refiner -f   # Refiner only, real-time
```

### Method 2: Direct Commands

```bash
cd ai-recruiter/agents/nlp/demo

# View all logs
cat backend.log

# Follow logs in real-time
tail -f backend.log

# Filter specific components
grep "SCORER" backend.log      # Scorer analysis
grep "REFINER" backend.log     # Question refinement
grep "FINAL" backend.log       # Final questions
grep "DECISION" backend.log    # Decision making
grep "MAIN LLM" backend.log    # Main LLM output

# Search for specific terms
grep "deepened" backend.log    # Questions that were deepened
grep "detailed" backend.log    # Detailed answers
grep "timeout" backend.log     # Timeouts or errors
```

---

## 📋 What You'll See

When a candidate answers in the demo, you'll see the complete pipeline log:

```
======================================================================
📊 PIPELINE LOG - Session: a1b2c3d4
======================================================================

🧠 SCORER - Analyzing candidate answer...
   ✓ Scorer completed in 450ms
   • Answer Quality: detailed
   • Engagement: engaged
   • Knowledge Level: demonstrated
   • Suggested Action: move_to_new_topic
   • Focus Area: scaling challenges with Neo4j
   • Knowledge Gaps: performance optimization, data modeling
   • Acknowledgment: "So you used Neo4j for the graph layer."

📋 DECISION - detailed (45w, quality=detailed)
   • Follow-up: No (move to new topic)
   • Instruction: [Start with: "So you used Neo4j..." | Move to: React]

🤖 MAIN LLM - Generating question...
   ✓ Generated in 650ms (42.3 tok/s)

   📝 Original Question:
   "So you used Neo4j for the graph layer. Tell me about your 
    React frontend implementation."

✨ REFINER - Enhancing question depth & specificity...
   ✓ Refinement completed in 520ms
   • Type: deepened
   • Improvements: Added specificity (state mgmt, scale), 
                   probes technical challenges

   ✏️  Refined Question:
   "So you used Neo4j for the graph layer. What specific challenges 
    did you face with React state management when scaling to thousands 
    of concurrent users?"

🔊 TTS - Synthesizing audio...
   ✓ Audio synthesized (2 segment(s))

======================================================================
✅ FINAL QUESTION: "So you used Neo4j for the graph layer. What 
   specific challenges did you face with React state management when 
   scaling to thousands of concurrent users?"
======================================================================
```

---

## 🎯 Common Use Cases

### 1. Watch Interview Progress
```bash
./view-logs.sh --final -f
```
Shows each final question as it's asked in real-time.

### 2. Monitor Scorer Quality
```bash
./view-logs.sh --scorer -f
```
See how well the scorer is analyzing answers.

### 3. Track Refinement Effectiveness
```bash
./view-logs.sh --refiner
```
Compare original vs refined questions to see improvements.

### 4. Debug Slow Responses
```bash
grep "completed in" backend.log
```
Find latency for each component.

### 5. Check for Errors
```bash
grep "✗\|⚠\|error\|timeout" backend.log
```
Find warnings and errors.

---

## 📊 Log Statistics

```bash
# Count refinement types
grep "Type:" backend.log | cut -d: -f3 | sort | uniq -c

# Average scorer latency
grep "Scorer completed" backend.log | grep -oP '\d+ms' | sed 's/ms//' | awk '{sum+=$1; n++} END {print sum/n "ms"}'

# Count questions asked
grep "FINAL QUESTION" backend.log | wc -l

# Most common answer qualities
grep "Answer Quality:" backend.log | cut -d: -f3 | sort | uniq -c | sort -rn
```

---

## 🧹 Log Management

### Clear Old Logs
```bash
> backend.log
> frontend.log
```
Or just restart the demo (logs are automatically cleared).

### Archive Logs
```bash
# Save logs from a session
cp backend.log interview_$(date +%Y%m%d_%H%M%S).log
```

### Rotate Logs
The logs are cleared each time you start the demo, so no rotation needed.

---

## 🚀 Quick Start

```bash
# Terminal 1: Start demo
cd ai-recruiter/agents/nlp/demo
./start-react-demo.sh

# Terminal 2: Watch logs
cd ai-recruiter/agents/nlp/demo
./view-logs.sh --follow

# Terminal 3: Monitor specific component
cd ai-recruiter/agents/nlp/demo
./view-logs.sh --refiner -f
```

---

## 💡 Tips

1. **Keep a terminal open** with `./view-logs.sh -f` while testing
2. **Compare original vs refined** to tune the refiner
3. **Watch scorer analysis** to verify it's working correctly
4. **Monitor latencies** to identify bottlenecks
5. **Check for errors** regularly during development

---

## 📞 Troubleshooting

**No logs appearing?**
- Check if demo is running: `ps aux | grep uvicorn`
- Check log file exists: `ls -l backend.log`
- Try: `tail -f backend.log`

**Logs too verbose?**
- Use filters: `./view-logs.sh --final`
- Or grep: `grep "FINAL" backend.log`

**Want to disable logging?**
- Comment out print statements in `agent.py`
- Or redirect to /dev/null (not recommended for dev)

---

Enjoy your enhanced visibility into the AI interview pipeline! 🎉
