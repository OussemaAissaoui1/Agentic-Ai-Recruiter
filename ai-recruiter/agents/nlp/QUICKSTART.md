# NLP Agent Quick Start Guide

## 📁 Files Created

```
agents/nlp/
├── __init__.py              # Package initialization
├── agent.py                 # ✅ Core NLP agent logic
├── vllm_engine.py           # ✅ vLLM inference engine wrapper
├── interview_state.py       # ✅ Interview state tracking
├── server.py                # ✅ FastAPI REST API server
├── requirements.txt         # ✅ Python dependencies
├── start_server.sh          # ✅ Server startup script
├── test_client.py           # ✅ Test client demo
├── README.md                # ✅ Comprehensive documentation
├── question_generator.py    # Legacy placeholder
└── agent_card.json          # A2A metadata
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd /teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter/agents/nlp
pip install -r requirements.txt
```

**Required packages:**
- fastapi
- uvicorn
- vllm
- transformers
- torch
- pydantic

### Step 2: Verify Model

Ensure the merged model exists at:
```
/teamspace/studios/this_studio/Agentic-Ai-Recruiter/model_cache
```

### Step 3: Start Server

```bash
# Option 1: Using startup script
./start_server.sh

# Option 2: Using startup script in dev mode (auto-reload)
./start_server.sh --reload

# Option 3: Direct uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000

# Option 4: Specific GPU
CUDA_VISIBLE_DEVICES=0 uvicorn server:app --host 0.0.0.0 --port 8000
```

### Step 4: Test the API

**Option A: Using the test client**
```bash
# In another terminal
python test_client.py
```

**Option B: Using cURL**
```bash
# Health check
curl http://localhost:8000/health

# Warmup
curl -X POST http://localhost:8000/warmup

# Generate question
curl -X POST http://localhost:8000/generate-question \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "session_id": "test-001",
  "candidate_cv": "Candidate: John Doe\nExperience: ML Engineer with GraphRAG",
  "job_role": "AI Engineering Intern",
  "conversation_history": [],
  "candidate_latest_answer": "Hello, I'm ready for the interview."
}
EOF
```

**Option C: Using Swagger UI**
```bash
# Open interactive API docs
open http://localhost:8000/docs
```

## 📊 Expected Output

When you start the server:
```
🚀 Starting NLP Agent server...
✅ NLP Agent initialized successfully
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

When you run the test client:
```
📊 Testing /health endpoint...
✅ Status: healthy
   Model loaded: True
   Warmed up: True

🔥 Testing /warmup endpoint...
✅ Warmup complete: warmup_complete

🎤 Starting Interview Simulation...
============================================================

Turn 1: Opening
------------------------------------------------------------
Recruiter: Hi Oussema! Thanks for joining me today. Tell me about your experience with GraphRAG at TALAN Tunisia?

[Metadata]
  Reasoning: opening_turn
  Turn: 1
  Topic depth: 0
  TTFT: 0.089s
  Latency: 0.342s
  Tokens/sec: 43.9
```

## 🔌 Integration with Other Agents

The NLP agent can be called by the Orchestrator agent or used standalone:

```python
# Example integration
import httpx

async def orchestrator_generate_question(session_id, candidate_cv, history, last_answer):
    """Called by orchestrator to get next question."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://nlp-agent:8000/generate-question",
            json={
                "session_id": session_id,
                "candidate_cv": candidate_cv,
                "job_role": "AI Engineering Intern",
                "conversation_history": history,
                "candidate_latest_answer": last_answer,
            }
        )
        return response.json()
```

## 🎯 Key Features Demonstrated

✅ **Adaptive Questioning** - Test client shows follow-up vs. new topic logic
✅ **Answer Analysis** - Specificity scoring and depth detection
✅ **Performance Metrics** - TTFT, latency, tokens/sec tracking
✅ **Session Management** - State tracking across conversation turns
✅ **Error Handling** - Recovery from termination attempts and duplicates

## 📖 Next Steps

1. **Read the full README.md** for detailed documentation
2. **Customize prompts** in `agent.py` → `SystemPromptBuilder`
3. **Tune parameters** like temperature, max_tokens, follow-up limits
4. **Integrate with Voice Agent** to get real candidate responses
5. **Connect to Scoring Agent** to use answer analysis metrics

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :8000

# Try different port
PORT=8001 ./start_server.sh
```

### Model not found
```bash
# Verify model exists
ls -la /teamspace/studios/this_studio/Agentic-Ai-Recruiter/model_cache/

# Update path in agent.py if needed
```

### GPU out of memory
```python
# In agent.py, reduce memory usage:
nlp_agent = NLPAgent(
    gpu_memory_utilization=0.85,  # Lower from 0.92
    max_model_len=1536,            # Lower from 2048
)
```

## 🎉 Success!

You now have a fully functional NLP agent that:
- Generates intelligent interview questions
- Analyzes candidate responses
- Manages conversation flow
- Provides performance metrics
- Handles edge cases gracefully

Check the **README.md** for comprehensive documentation!
