# Enhanced Logging Guide

## 🎯 What You'll See

Every time a candidate answers, you'll get a **detailed pipeline log** showing exactly what happens at each step.

---

## 📋 Log Format

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
   • Instruction: [Start with: "So you used Neo4j..." | Move to: React Frontend]

🤖 MAIN LLM - Generating question...
   ✓ Generated in 650ms (42.3 tok/s)
   
   📝 Original Question:
   "So you used Neo4j for the graph layer. Tell me about your React 
    frontend implementation."

✨ REFINER - Enhancing question depth & specificity...
   ✓ Refinement completed in 520ms
   • Type: deepened
   • Improvements: Added specificity (state management, scale), 
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

## 🔍 What Each Section Shows

### 🧠 SCORER
- **What it does**: Qwen 1.5B analyzes the candidate's answer
- **Key info**:
  - Answer quality (vague/partial/detailed/off_topic)
  - Candidate engagement level
  - Knowledge demonstrated
  - What to do next (follow-up or move on)
  - Knowledge gaps to probe
  - Natural acknowledgment phrase

### 📋 DECISION
- **What it does**: State tracker decides the strategy
- **Key info**:
  - Should we follow up or move to new topic?
  - Instruction sent to main LLM
  - Reasoning for the decision

### 🤖 MAIN LLM
- **What it does**: Fine-tuned Llama generates the question
- **Key info**:
  - Original question (before refinement)
  - Generation speed and latency
  - Shows the baseline output

### ✨ REFINER
- **What it does**: Qwen 1.5B enhances the question
- **Key info**:
  - Refinement type (deepened/specified/rephrased/maintained)
  - What improvements were made
  - The refined question
  - **This is where generic → specific happens!**

### 🔊 TTS
- **What it does**: Synthesizes speech
- **Key info**:
  - Audio generation status
  - Number of segments

### ✅ FINAL QUESTION
- The actual question delivered to the candidate
- After all processing steps

---

## 💡 How to Use These Logs

### 1. **Debugging**
```bash
# If questions are bad, check which step failed:
- Scorer giving bad analysis? → Check scorer model
- Main LLM generating generic? → Check prompts/instructions  
- Refiner not improving? → Check refiner prompts
- Slow responses? → Check latency numbers
```

### 2. **Monitoring**
Track these metrics:
- Refinement rate: How often does refiner change questions?
- Scorer confidence: Are analyses accurate?
- Latency breakdown: Which component is slowest?
- Error rate: How often do timeouts/errors occur?

### 3. **Quality Assurance**
Compare original vs refined:
```
📝 Original: "Tell me about your React frontend."
✏️  Refined: "What specific challenges did you face with React state 
             management when scaling to thousands of concurrent users?"
```
See the improvement in real-time!

### 4. **Optimization**
Identify patterns:
- Which topics trigger follow-ups?
- What knowledge gaps are commonly found?
- Which acknowledgments work best?
- Where are the bottlenecks?

---

## 🛠️ Log Management

### Save logs to file:
```bash
cd ai-recruiter/agents/nlp
python server.py 2>&1 | tee nlp_agent.log
```

### Filter specific components:
```bash
# Only scorer analysis
python server.py 2>&1 | grep "SCORER"

# Only refiner output
python server.py 2>&1 | grep "REFINER"

# Only final questions
python server.py 2>&1 | grep "FINAL QUESTION"

# Only errors/warnings
python server.py 2>&1 | grep "✗\|⚠"
```

### Disable logs for production:
Comment out the logging lines in `agent.py` (lines 360-450) or redirect to `/dev/null`:
```bash
python server.py > /dev/null 2>&1
```

---

## 📊 Example Analysis Session

### Scenario: Debugging why questions are too generic

**Step 1: Check scorer output**
```
🧠 SCORER - Analyzing candidate answer...
   • Question Focus: [empty or vague]  ← PROBLEM!
```
→ Scorer isn't identifying specific areas to probe

**Step 2: Check main LLM output**
```
📝 Original Question:
   "Tell me more about that project."  ← Generic!
```
→ Main LLM generating generic without good focus

**Step 3: Check refiner**
```
✨ REFINER - Enhancing question depth & specificity...
   • Type: maintained
   • Improvements: timeout  ← PROBLEM!
```
→ Refiner timing out, can't fix generic question

**Solution**: Increase refiner timeout or optimize scorer prompts

---

## 🎨 Log Symbols Reference

- ✅ Success / Final output
- ✓ Component completed successfully
- ✗ Error occurred
- ⚠ Warning / Retry
- 📊 Pipeline overview
- 🧠 Scorer (analysis)
- 📋 Decision making
- 🤖 Main LLM (generation)
- ✨ Refiner (enhancement)
- 🔊 TTS (audio synthesis)
- 📝 Original output
- ✏️ Refined output
- • Bullet point / detail

---

## 🚀 Quick Start

1. **Start your NLP agent**:
   ```bash
   cd ai-recruiter/agents/nlp
   ./start_server.sh
   ```

2. **Run an interview session** (logs appear automatically)

3. **Watch the pipeline in action**:
   - See scorer analyze answers
   - Watch main LLM generate questions
   - Observe refiner enhance them
   - Compare original vs refined

4. **Analyze patterns** and tune parameters

---

## 📈 What Good Logs Look Like

```
✅ Scorer completing in 300-500ms
✅ Main LLM generating in 400-800ms (>30 tok/s)
✅ Refiner completing in 400-800ms
✅ Refinement type: "deepened" or "specified" (not "maintained")
✅ Original vs Refined showing clear improvements
✅ No errors or timeouts
```

## 🔴 Warning Signs

```
⚠ Scorer timing out or erroring frequently
⚠ Main LLM retries needed often
⚠ Refiner always returning "maintained" (not refining)
⚠ Latencies > 2 seconds for any component
⚠ Generic questions in Original output
⚠ No improvement in Refined output
```

---

## 💡 Pro Tips

1. **Compare sessions**: Save logs from good and bad interviews, compare patterns

2. **Track refinement rate**: 
   ```bash
   grep "Type:" nlp_agent.log | sort | uniq -c
   ```

3. **Monitor latency**:
   ```bash
   grep "completed in" nlp_agent.log
   ```

4. **Find problem areas**:
   ```bash
   grep "✗\|⚠\|timeout\|error" nlp_agent.log
   ```

5. **Extract final questions only**:
   ```bash
   grep "FINAL QUESTION" nlp_agent.log | sed 's/.*FINAL QUESTION: //'
   ```

---

## 🎯 Summary

You now have **complete visibility** into the 3-LLM pipeline:
- See exactly what the scorer analyzed
- Compare original vs refined questions
- Monitor performance and quality
- Debug issues quickly
- Optimize based on data

The logs are **live** and will appear automatically when you run interviews! 🎉
