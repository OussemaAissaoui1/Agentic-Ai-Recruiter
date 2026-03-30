# NLP Agent - AI Recruiter Interview Conversation Manager

**Version:** 1.0.0
**Purpose:** Manages natural language interactions during AI-powered recruitment interviews using a fine-tuned Llama 3.1 8B recruiter persona model.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [API Endpoints](#api-endpoints)
5. [Usage Examples](#usage-examples)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Advanced Features](#advanced-features)

---

## 🎯 Overview

The NLP Agent is responsible for conducting intelligent, contextual interview conversations. It:

- **Generates interview questions** based on candidate CV and job role
- **Analyzes candidate responses** for depth and specificity
- **Manages interview flow** by deciding when to ask follow-ups vs. move to new topics
- **Prevents repetition** and handles edge cases (termination attempts, skip requests)
- **Provides metrics** for latency, token usage, and answer quality

### Key Features

✅ **Adaptive Questioning** - Asks follow-ups for vague answers, moves on when satisfied
✅ **CV-Aware** - Asks specific questions grounded in the candidate's actual experience
✅ **Fast Inference** - vLLM with chunked prefill, prefix caching, and CUDA graphs
✅ **Topic Depth Tracking** - Prevents getting stuck, limits follow-ups per topic
✅ **Duplicate Prevention** - Tracks asked questions to avoid repeating
✅ **Recovery Mechanisms** - Handles model failures gracefully

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NLP Agent                                │
│                                                                 │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  FastAPI       │  │  NLPAgent       │  │  vLLMRecruiter  │ │
│  │  Server        │─→│  (Core Logic)   │─→│  Engine         │ │
│  │  (server.py)   │  │  (agent.py)     │  │  (vllm_engine)  │ │
│  └────────────────┘  └─────────────────┘  └─────────────────┘ │
│                             │                                   │
│                             ↓                                   │
│                    ┌─────────────────┐                          │
│                    │ InterviewState  │                          │
│                    │ Tracker         │                          │
│                    │ (interview_state)│                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │  Fine-tuned     │
                    │  Llama 3.1 8B   │
                    │  Recruiter Model│
                    └─────────────────┘
```

### Data Flow

1. **Client sends request** → FastAPI server (`POST /generate-question`)
2. **NLPAgent analyzes** candidate's latest answer using `InterviewStateTracker`
3. **StateTracker decides** → Follow-up on same topic? Or move to new topic?
4. **SystemPromptBuilder** creates CV-aware, randomized system prompt
5. **VLLMRecruiterEngine** generates question using fine-tuned model
6. **Validation** checks for termination attempts, duplicates
7. **Response returned** with question + metadata (reasoning, metrics, topic depth)

---

## 🧩 Components

### File Structure

| File | Purpose |
|------|---------|
| `server.py` | FastAPI REST API server |
| `agent.py` | Core NLP agent logic and orchestration |
| `vllm_engine.py` | vLLM async inference engine wrapper |
| `interview_state.py` | Interview progression and state tracking |
| `question_generator.py` | Legacy placeholder (replaced by agent.py) |
| `agent_card.json` | A2A discovery metadata |
| `requirements.txt` | Python dependencies |

### 1. **server.py** - FastAPI REST API

Exposes HTTP endpoints for:
- Question generation
- Response analysis
- Session management
- Health checks

**Key endpoints:**
- `POST /generate-question` - Main endpoint for conversation
- `POST /analyze-response` - Get answer quality metrics
- `GET /session/{id}/stats` - Session statistics
- `POST /warmup` - Pre-load model caches

### 2. **agent.py** - Core NLP Agent Logic

The orchestrator that:
- Manages interview sessions (state mapping)
- Coordinates between state tracker and vLLM engine
- Builds system prompts (randomized or compact)
- Validates and cleans model outputs
- Provides high-level API for question generation

**Classes:**
- `NLPAgent` - Main agent class
- `QuestionGenerationRequest/Response` - API data models
- `SystemPromptBuilder` - Prompt engineering

### 3. **vllm_engine.py** - vLLM Inference Wrapper

High-performance async wrapper around vLLM's `AsyncLLMEngine`.

**Optimizations:**
- ✅ **Chunked prefill** - Reduces TTFT (time to first token) on long prompts
- ✅ **Prefix caching** - Reuses system prompt KV cache across turns
- ✅ **CUDA graphs** - Lower per-token latency
- ✅ **Streaming** - Real TTFT measurement

**Classes:**
- `VLLMRecruiterEngine` - Async engine wrapper
- `GenerationMetrics` - Performance metrics dataclass

**Methods:**
- `warmup()` - Prime KV cache with system prompt
- `generate()` - Generate question with streaming
- `_clean_response()` - Remove special tokens, multi-turn leakage

### 4. **interview_state.py** - Interview State Management

Tracks interview progression and decides follow-up behavior.

**Logic:**
- **Vague answer** (< 25 words, low specificity) → Follow-up on same topic
- **Detailed answer** (≥ 30 words + specifics) → Move to next topic
- **Skip request** detected → Acknowledge and move on
- **Max follow-ups reached** (3) → Force move to new topic

**Classes:**
- `InterviewStateTracker` - Main state tracker
- `AnswerAnalysis` - Answer quality metrics

**Key methods:**
- `analyze_answer()` - Compute word count, specificity score, etc.
- `should_follow_up()` - Decision logic: follow-up or move on?
- `get_instruction()` - Generate instruction for model
- `record_question()` - Track asked questions
- `is_duplicate_question()` - Prevent repeats (>80% word overlap)

---

## 🔌 API Endpoints

### POST `/generate-question`

Generate the next interview question based on conversation context.

**Request:**
```json
{
  "session_id": "interview-123",
  "candidate_cv": "Candidate: Oussema Aissaoui\nRole: AI Engineering Intern\n...",
  "job_role": "AI Engineering Intern",
  "conversation_history": [
    {
      "candidate": "I worked on GraphRAG at TALAN Tunisia.",
      "recruiter": "What was your specific role in the GraphRAG project?"
    }
  ],
  "candidate_latest_answer": "I implemented the LLM reasoning pipeline.",
  "max_tokens": 48,
  "temperature": 0.6
}
```

**Response:**
```json
{
  "question": "What technical challenges did you face when implementing the LLM reasoning pipeline?",
  "reasoning": "short_no_specifics (18 words, score=1)",
  "topic_depth": 1,
  "turn_count": 2,
  "is_follow_up": true,
  "metrics": {
    "latency_sec": 0.342,
    "ttft_sec": 0.089,
    "num_tokens": 15,
    "tokens_per_sec": 43.86
  }
}
```

### POST `/analyze-response`

Analyze a candidate's answer for quality signals.

**Request:**
```json
{
  "session_id": "interview-123",
  "candidate_answer": "I used Neo4j for the graph database and implemented RAG using Llama 3.1."
}
```

**Response:**
```json
{
  "word_count": 14,
  "sentence_count": 1,
  "specificity_score": 3,
  "has_specifics": true,
  "has_numbers": true,
  "is_detailed": false,
  "wants_to_skip": false,
  "is_nonsense": false
}
```

### GET `/session/{session_id}/stats`

Get interview session statistics.

**Response:**
```json
{
  "turn_count": 5,
  "topic_depth": 2,
  "consecutive_vague": 0,
  "questions_asked_count": 5,
  "topics_covered": ["GraphRAG", "YOLOv8"]
}
```

---

## 📚 Usage Examples

### Python Client Example

```python
import httpx
import asyncio

async def conduct_interview():
    base_url = "http://localhost:8000"
    session_id = "interview-001"

    # Warm up model
    await httpx.post(f"{base_url}/warmup")

    # Candidate CV (simplified)
    cv = """
    Candidate: Oussema Aissaoui
    Role: AI Engineering Intern

    Experience:
    - GraphRAG + LLM reasoning at TALAN Tunisia
    - YOLOv8 + SAM blood cell segmentation
    - ML model for crop water efficiency
    """

    conversation_history = []
    candidate_answer = "Hello! I'm ready for the interview."

    for turn in range(5):
        # Generate next question
        response = await httpx.post(
            f"{base_url}/generate-question",
            json={
                "session_id": session_id,
                "candidate_cv": cv,
                "job_role": "AI Engineering Intern",
                "conversation_history": conversation_history,
                "candidate_latest_answer": candidate_answer,
            },
        )

        data = response.json()
        question = data["question"]

        print(f"\nRecruiter: {question}")
        print(f"[{data['reasoning']}]")

        # Simulate candidate answer
        candidate_answer = input("Candidate > ")

        # Update history
        conversation_history.append({
            "candidate": candidate_answer,
            "recruiter": question,
        })

    # End session
    await httpx.delete(f"{base_url}/session/{session_id}")

asyncio.run(conduct_interview())
```

### cURL Example

```bash
# Generate question
curl -X POST http://localhost:8000/generate-question \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-123",
    "candidate_cv": "Candidate: John Doe\nExperience: ML Engineer, 3 years",
    "job_role": "AI Engineering Intern",
    "conversation_history": [],
    "candidate_latest_answer": "Hello, I am ready."
  }'
```

---

## 🚀 Installation

### 1. Install Dependencies

```bash
cd /teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter/agents/nlp
pip install -r requirements.txt
```

### 2. Verify Model Path

The agent expects the merged model at:
```
/teamspace/studios/this_studio/Agentic-Ai-Recruiter/ai-recruiter/ml/recruiter-persona/training/output_v3/recruiter-persona-llama-3.1-8b/merged-full
```

### 3. Run Server

```bash
# Production (uvicorn)
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

# Development (with reload)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# With specific GPU
CUDA_VISIBLE_DEVICES=0 uvicorn server:app --host 0.0.0.0 --port 8000
```

### 4. Test Endpoints

```bash
# Swagger UI (interactive docs)
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/health
```

---

## ⚙️ Configuration

### Model Configuration (agent.py)

```python
nlp_agent = NLPAgent(
    model_path="/path/to/merged/model",  # Local path or HF Hub
    dtype="bfloat16",                     # bfloat16 or float16
    max_model_len=2048,                   # Max sequence length
    gpu_memory_utilization=0.92,          # Fraction of GPU to use
    use_compact_prompt=False,             # True = lower latency, less detail
)
```

### Generation Parameters

```python
response = await nlp_agent.generate_question(
    request=request,
    max_tokens=48,              # Max tokens per question
    temperature=0.6,            # Higher = more creative (0.0 - 2.0)
    top_p=0.85,                 # Nucleus sampling threshold
    repetition_penalty=1.15,    # Penalty for repeating tokens
)
```

---

## 🔬 Advanced Features

### 1. **Answer Depth Analysis**

The state tracker uses keyword-based specificity scoring:

**Specificity keywords:**
- Action verbs: `"i implemented"`, `"i designed"`, `"i trained"`
- Technical terms: `"tensorflow"`, `"pytorch"`, `"graphrag"`, `"neo4j"`
- Metrics: `"accuracy"`, `"performance"`, `"reduced"`, `"improved"`
- Problem-solving: `"the challenge"`, `"the problem was"`, `"resulted in"`

**Example scoring:**
```
"I worked on ML" → 0 keywords → vague
"I trained a YOLOv8 model for segmentation" → 3 keywords → specific
```

### 2. **Adaptive Follow-up Hints**

The agent cycles through different types of follow-up questions:

1. **Specific role** - "What did YOU personally do?"
2. **Technical challenge** - "What challenge did you face and how did you solve it?"
3. **Measurable results** - "What were the numbers/metrics/improvements?"
4. **Technical walkthrough** - "Walk me through your approach step-by-step."
5. **Technology choices** - "Why did you choose that technology?"
6. **Testing/validation** - "How did you test or validate your solution?"

### 3. **Performance Metrics**

Every response includes:
- `latency_sec` - Total generation time
- `ttft_sec` - Time to first token (TTFT)
- `num_tokens` - Tokens generated
- `tokens_per_sec` - Throughput

Typical performance on L4 GPU:
- TTFT: ~80-150ms
- Throughput: ~40-50 tok/s
- Latency (48 tokens): ~300-400ms

---

## 📖 References

- Fine-tuned model training: `/ml/recruiter-persona/training/`
- Training script: `/ml/recruiter-persona/training/train.py`
- Test inference: `/ml/recruiter-persona/training/scripts/test_merged_vllm.py`
- vLLM docs: https://docs.vllm.ai/
- FastAPI docs: https://fastapi.tiangolo.com/

---

**Built with:**
- 🦙 Llama 3.1 8B (fine-tuned recruiter persona)
- ⚡ vLLM (async inference engine)
- 🚀 FastAPI (REST API)
- 🧠 Custom state tracking logic
